#include "ExtractTileOperations.h"

#include "CanonicalizeGPUVars.h"
#include "EqSatIRParser.h"
#include "EqSatIRPrinter.h"
#include "IRMatch.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "Simplify.h"
#include "Util.h"
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <memory>
#include <regex>
#include <sstream>
#include <string>
#include <sys/fcntl.h>
#include <unistd.h>
#include <unordered_set>
#include <utility>

/** \file Support extraction of AMX instructions. */

/**
 * https://asciiflow.com/#/share/eJyVUkFugzAQ%2FMrKxwoRhdAkza23SmlySHvogQsBp7FkbGSbAoryiz6nr%2BlLugZDk6ghKvJhbXZmd2b3QEScUbIQBece4XFNFVmQQ0SqiCwegtCLSI1RMBtjZGhl8BIRAHh%2BeoFVbBSr4Pq36ZOiSOBpX5cDCEikSGhuipjzun0pmdnD4%2BqtwX9%2Ffg2cLmUcTML76WyO4VAtWJ%2Ff7kIkWMEJ6gbBae2%2F3q53OHBuFBz3TS1HodPqfvUO3%2F4wO7gQag07IXqVkCuZU4VzyApuWI5BAJkdZ0K1B2ZP2%2BwJ%2FEs%2BjhKY0EYViWFSaMAaO6kypBY1hLCtDRIvMTvsekmlsc2kiGgKMw2cxqkGIyEGjn%2FlzonoIMjPUibeQX5Q1bHGisbav%2FBh2kHW2ESzdlaZkqUltaFd9UZ25TnIrIOg%2Bb7vQykLnv661GysRSaSF1k78HkHcaSbntSReLAtTL%2FscOlaI9rxYaRzzgwUOTrZeOCokLzN0TDqRYvUqtFwB6Fvqco9S5r%2BBCiqsWmNLHabzny2Y7E4PyJHcvwBx0t%2BJw%3D%3D)
 *
 *   LHS Matrix                           RHS Matrix
 *
 *      K                            conceptually      with AMX
 *  ┌────────┐
 *  │12345678│                             N             N*4
 *M │        │                            ┌──┐        ┌────────┐
 *  └────────┘                            │1 │     K/4│1234    │
 *                                        │2 │        │5678    │
 * To properly multiply 2 matrices, the   │3 │        └────────┘
 * AMX instructions perform many 4 byte  K│4 │
 * dot products, this leads to a lot of   │5 │
 * striding over 4 byte areas.            │6 │
 * Normally the row of the LHS matrix,    │7 │
 * 123... would multiply with the column  │8 │
 * of the RHS matrix 123..., but with AMX └──┘
 * this column is split up into a matrix of columns / 4 byte and rows * 4.
 * which then results in K/4 dot products per row.
 *
 */

namespace Halide {
namespace Internal {
using std::string;
using std::vector;

namespace {
template<int Dim>
struct Tile {
    bool result;
    Expr base;
    Expr stride[Dim];
    int extent[Dim];
};

enum class AMXOpType {
    Int8,
    Bfloat16,
};

/// returns the appropriate `Halide::Type` for the given operation type
Type amx_op_type_result_type(AMXOpType op_ty) {
    switch (op_ty) {
    case AMXOpType::Int8:
        return Int(32, 256);
    case AMXOpType::Bfloat16:
        return Float(32, 256);
    default:
        internal_error << "Unexpected";
        return Type();
    }
}

int amx_op_type_size(AMXOpType op_ty) {
    switch (op_ty) {
    case AMXOpType::Int8:
        return 1;
    case AMXOpType::Bfloat16:
        return 2;
    default:
        internal_error << "Unexpected";
        return -1;
    }
}

const auto wild_i32 = Variable::make(Int(32), "*");
const auto wild_i32x = Variable::make(Int(32, 0), "*");

Tile<1> get_1d_tile_index(const Expr &e) {
    if (const auto *r1 = e.as<Ramp>()) {
        return {true, r1->base, {r1->stride}, {r1->lanes}};
    }

    return {};
}

Tile<2> get_2d_tile_index(const Expr &e) {
    // ramp(ramp(base, 1, 4), x4(stride), 4)
    vector<Expr> matches;
    if (const auto *r1 = e.as<Ramp>()) {
        if (const auto *r2 = r1->base.as<Ramp>()) {
            auto ramp_2d_pattern = Ramp::make(Ramp::make(wild_i32, wild_i32, r2->lanes), Broadcast::make(wild_i32, r2->lanes), r1->lanes);
            if (expr_match(ramp_2d_pattern, e, matches)) {
                return {true, std::move(matches[0]), {std::move(matches[2]), std::move(matches[1])}, {r1->lanes, r2->lanes}};
            }
        }
    }
    return {};
}

Tile<3> get_3d_tile_index(const Expr &e) {
    vector<Expr> matches;

    // there could be a sub node
    const Sub *sub = e.as<Sub>();
    const Add *add = nullptr;

    if (sub) {
        add = sub->a.as<Add>();
    } else {
        add = e.as<Add>();
    }

    if (!add) {
        return {};
    }

    const auto &first = add->a;
    const auto &second = add->b;

    // ramp(x[x*r](base), x[x*r](stride), x) + x[x*y](ramp(idx, 1, r))

    const auto *r1 = first.as<Ramp>();
    const auto *b2 = second.as<Broadcast>();
    if (!r1 && !b2) {
        // Try switching the order
        r1 = second.as<Ramp>();
        b2 = first.as<Broadcast>();
    }
    if (!r1 || !b2) {
        return {};
    }

    const auto *b1 = r1->base.as<Broadcast>();
    const auto *r2 = b2->value.as<Ramp>();

    if (!b1 || !r2) {
        return {};
    }

    int x_tile = r1->lanes;
    int r_tile = r2->lanes;
    int y_tile = b1->lanes / r_tile;
    if (y_tile != b2->lanes / x_tile) {
        return {};
    }

    auto pattern1 = Ramp::make(Broadcast::make(wild_i32, b1->lanes), Broadcast::make(wild_i32, b1->lanes), r1->lanes);
    if (!expr_match(pattern1, first, matches)) {
        return {};
    }
    Expr base = std::move(matches[0]);
    Expr x_stride = std::move(matches[1]);

    auto pattern2 = Broadcast::make(Ramp::make(wild_i32, wild_i32, r2->lanes), b2->lanes);
    if (!expr_match(pattern2, second, matches)) {
        return {};
    }
    base += std::move(matches[0]);
    Expr r_stride = std::move(matches[1]);

    if (sub) {
        Expr adj = sub->b;
        const Broadcast *bcast = adj.as<Broadcast>();

        if (!bcast) {
            return {};
        }

        if (bcast->lanes != b1->lanes * r1->lanes) {
            return {};
        }

        base -= bcast->value;
    }

    return {true, base, {x_stride, 0, r_stride}, {x_tile, y_tile, r_tile}};
}

/**
 * \brief Get the 3d rhs tile index configuration
 *
 * \param e index expression
 * \param element_width the width of the elements, 1 for u8/i8, 2 for bf16
 * \return Tile<3> the tile configuration found
 *
 * The pattern which is getting matched looks roughly like
 * `broadcast(ramp(0, 1, r), x*y) / broadcast(4, x*y*r) + optional(broadcast(base, x*y*r)) * broadcast(8, x*y*r) +
 *  broadcast(ramp(0, 1, r), x*y) % broadcast(4, x*y*r) +
 *  broadcast(ramp(broadcast(_, r), broadcast(4, r), y) , x)`
 */
Tile<3> get_3d_rhs_tile_index(const Expr &e, int element_width) {
    const auto *sub = e.as<Sub>();
    const Add *add_lhs = nullptr;

    // there's not always a sub pattern
    // This depends on whether we have an ImageParam or a Buffer
    if (!sub) {
        add_lhs = e.as<Add>();
    } else {
        add_lhs = sub->a.as<Add>();
    }

    if (!add_lhs) {
        return {};
    }

    // The right hand side of the add expression is used for retrieving the dimensions of the matrix.
    // obtain the x, y, r dimensions
    // this expr looks like below, the shape of `add_lhs->a` can be seen further down below
    // broadcast(ramp(0, 1, r), x*y) % broadcast(4, x*y*r) + broadcast(ramp(broadcast(base, r), broadcast(4, r), y) , x)
    const Add *dim_expr = add_lhs->b.as<Add>();

    if (!dim_expr) {
        return {};
    }

    // broadcast(ramp(broadcast(_, r), broadcast(4, r), y), x)
    const Broadcast *base_stride_bc = dim_expr->b.as<Broadcast>();

    if (!base_stride_bc) {
        return {};
    }

    int tile_x = base_stride_bc->lanes;

    // broadcast(ramp(0, 1, r), x*y) % broadcast(4, x*y*r)
    std::vector<Expr> results{};
    const Expr mod_pattern = Mod::make(wild_i32x, Broadcast::make(4 / element_width, 0));
    if (!expr_match(mod_pattern, dim_expr->a, results)) {
        return {};
    }

    // broadcast(ramp(0, 1, r), x*y)
    const Broadcast *bc_ramp = results[0].as<Broadcast>();

    if (!bc_ramp) {
        return {};
    }

    int tile_xy = bc_ramp->lanes;
    int tile_y = tile_xy / tile_x;

    // ramp(0, 1, r)
    const Ramp *r_ramp = bc_ramp->value.as<Ramp>();

    if (!r_ramp) {
        return {};
    }

    int tile_r = r_ramp->lanes;

    // get the base and stride
    // ramp(broadcast(_, r), broadcast(4, r), y)
    const Expr base_stride_ramp_pattern = Ramp::make(Broadcast::make(wild_i32, tile_r), Broadcast::make(4 / element_width, tile_r), tile_y);
    if (!expr_match(base_stride_ramp_pattern, base_stride_bc->value, results)) {
        return {};
    }

    Expr base = results[0];
    Expr stride;

    bool found_stride = false;

    // the following pattern will match the following shape
    // broadcast(ramp(0, 1, k), x*y) / broadcast(4, x*y*k) * broadcast(_, x*y*k)
    // where the stride is marked by _.

    // this stride pattern can occur if `tile_r` is the same size as `acc`
    auto stride_pattern = Broadcast::make(Ramp::make(0, 1, tile_r), tile_x * tile_y) / Broadcast::make((4 / element_width), tile_x * tile_y * tile_r) * Broadcast::make(wild_i32, tile_x * tile_y * tile_r);

    if (expr_match(stride_pattern, add_lhs->a, results)) {
        found_stride = true;
        stride = std::move(results[0]);
    }

    // This pattern is similar to the above except with an additional offset to iterate over the tiles in the k dimension
    // (broadcast(ramp(0, 1, k), m * n) / broadcast(4, m*n*k) + _) * broadcast(_, m*n*k)
    // here the first _ marks the base and the second _ the stride.
    if (!found_stride) {
        stride_pattern = (Broadcast::make(Ramp::make(0, 1, tile_r), tile_x * tile_y) / Broadcast::make((4 / element_width), tile_x * tile_y * tile_r) + wild_i32) * Broadcast::make(wild_i32, tile_x * tile_y * tile_r);
        if (expr_match(stride_pattern, add_lhs->a, results)) {
            found_stride = true;
            stride = std::move(results[1]);
            base = std::move(results[0]) * stride + base;
        }
    }

    if (!found_stride) {
        return {};
    }

    return {true, base, {stride, 0, 0}, {tile_x, tile_y, tile_r}};
}

struct BaseStride {
    bool result{false};
    Expr base{};
    Expr stride{};
};

BaseStride get_rhs_tile_index(const Expr &index, int element_width, int tile_x, int tile_y, int tile_r) {
    const auto rhs_tile2 = get_2d_tile_index(index);

    if (!rhs_tile2.result) {
        const auto rhs_tile1 = get_1d_tile_index(index);

        if (!rhs_tile1.result) {
            auto rhs_tile3 = get_3d_rhs_tile_index(index, element_width);
            if (rhs_tile3.extent[0] != tile_x || rhs_tile3.extent[1] != tile_y || rhs_tile3.extent[2] != tile_r) {
                return {};
            }

            return {true, rhs_tile3.base, rhs_tile3.stride[0] * element_width};
        } else {
            // 1D: degenerate as dot product. There are two cases:
            //   * tile_r is 4, so effectively there is only one row in the loaded tile
            //   * rhs.stride.1 == 4 && tile_y = 1, where the loaded RHS has shape (K/4)x4
            //     and is contiguous in the memory
            if (rhs_tile1.extent[0] != tile_y * tile_r) {
                return {};
            }
            if (!(rhs_tile1.stride[0].as<IntImm>() && rhs_tile1.stride[0].as<IntImm>()->value == 1)) {
                return {};
            }

            if (tile_r == 4 / element_width) {
                return {true, rhs_tile1.base, 0};
            }

            if (tile_y == 1) {
                // 4 elements in u8/i8 and 2 elements for bf16.
                return {true, rhs_tile1.base, 4 / element_width};
            }

            return {};
        }
    } else {
        // The only case where there is a ramp of ramp is when tile_y = 1 and so RHS has size (K/4)x4
        // (and rhs.stride.1 != 4, for o.w. it degenerates to 1D)
        if (tile_y != rhs_tile2.extent[0] || tile_r != rhs_tile2.extent[1]) {
            return {};
        }
        if (!(rhs_tile2.stride[1].as<IntImm>() && rhs_tile2.stride[1].as<IntImm>()->value == 1)) {
            return {};
        }

        if (tile_y != 1) {
            return {};
        }

        return {true, rhs_tile2.base, rhs_tile2.stride[0]};
    }
}

struct Matmul {
    bool result = false;
    Stmt stmt;
    int tile_x;
    int tile_y;
    int tile_r;
};

Matmul convert_to_matmul(const Store *op, const string &new_name, AMXOpType op_type) {
    // m[ramp(0, 1, S)] = VectorAdd(lhs[{XYR tile}] * xX(rhs[{YR tile}])) + m[ramp(0, 1, S)]
    const auto wild_i8x = Variable::make(Int(8, 0), "*");
    const auto wild_u8x = Variable::make(UInt(8, 0), "*");
    const auto wild_bf16x = Variable::make(BFloat(16, 0), "*");
    const auto wild_f32x = Variable::make(Float(32, 0), "*");

    vector<Expr> matches;
    if (op_type == AMXOpType::Int8) {
        const auto pattern1 = wild_i32x + wild_i32x;
        if (!expr_match(pattern1, op->value, matches)) {
            return {};
        }
    } else {  // AMXOpType::Bfloat16
        const auto pattern1 = wild_f32x + wild_f32x;
        if (!expr_match(pattern1, op->value, matches)) {
            return {};
        }
    }

    const auto *reduce = matches[0].as<VectorReduce>();
    const auto *load = matches[1].as<Load>();
    if (!reduce || reduce->op != VectorReduce::Add) {
        return {};
    }
    if (!load || load->name != op->name || !equal(load->index, op->index)) {
        return {};
    }

    if (op_type == AMXOpType::Int8) {
        auto pattern2 = cast(Int(32, 0), cast(Int(32, 0), wild_i8x) * wild_i32x);
        auto pattern2_unsigned = cast(Int(32, 0), cast(Int(32, 0), wild_u8x) * wild_i32x);

        if (!(expr_match(pattern2, reduce->value, matches) || expr_match(pattern2_unsigned, reduce->value, matches))) {
            return {};
        }
    } else {
        auto pattern2 = cast(Float(32, 0), cast(Float(32, 0), wild_bf16x) * wild_f32x);

        if (!expr_match(pattern2, reduce->value, matches)) {
            return {};
        }
    }

    const auto *lhs_load = matches[0].as<Load>();
    const auto *rhs_broadcast = matches[1].as<Broadcast>();

    const Cast *rhs_cast = nullptr;

    if (lhs_load && !rhs_broadcast) {
        // now working on a larger k dimension
        // with a K dimension of 4 (or 2) with bf16 all the elements in the right-hand matrix are
        // layed out in a way that multiplying with a column can be done in a single dot product.
        // Therefore the indexing can be reused with a broadcast,
        // with higher K dimensions this can no longer be done and the broadcast won't exist.
        // ┌──┐
        // │1 │
        // │2 │
        // │3 │   ┌────────┐
        // │4 │   │1234    │
        // │5 │   │5678    │
        // │6 │   └────────┘
        // │7 │
        // │8 │
        // └──┘
        rhs_cast = matches[1].as<Cast>();
    } else {
        rhs_cast = rhs_broadcast->value.as<Cast>();
    }

    if (!lhs_load || !rhs_cast) {
        return {};
    }

    if (rhs_cast) {
        bool is_i8_u8 = rhs_cast->value.type().element_of() == Int(8) || rhs_cast->value.type().element_of() == UInt(8);
        bool is_bf16 = rhs_cast->value.type().element_of() == BFloat(16);

        if ((op_type == AMXOpType::Int8 && !is_i8_u8) || (op_type == AMXOpType::Bfloat16 && !is_bf16)) {
            user_error << "Expected rhs type of " << (op_type == AMXOpType::Int8 ? "i8/u8" : "bf16")
                       << ", got " << rhs_cast->value.type() << " instead.\nIn Expression: " << Expr(rhs_cast);
        }
    } else {
        return {};
    }

    const auto *rhs_load = rhs_cast->value.as<Load>();
    if (!rhs_load) {
        return {};
    }

    const auto lhs_tile = get_3d_tile_index(lhs_load->index);

    if (!lhs_tile.result) {
        return {};
    }

    const int tile_x = lhs_tile.extent[0];
    const int tile_y = lhs_tile.extent[1];
    const int tile_r = lhs_tile.extent[2];
    const int factor = reduce->value.type().lanes() / reduce->type.lanes();

    Expr rhs_base;
    Expr rhs_stride;

    auto opt_base_stride = get_rhs_tile_index(rhs_load->index, amx_op_type_size(op_type), tile_x, tile_y, tile_r);

    if (!opt_base_stride.result) {
        return {};
    }

    rhs_base = opt_base_stride.base;
    rhs_stride = opt_base_stride.stride;

    if (op->index.type().lanes() != tile_x * tile_y ||
        factor != tile_r) {
        return {};
    }

    // {rows, colbytes, var, index}
    auto lhs_var = Variable::make(Handle(), lhs_load->name);
    const auto &lhs_load_type = lhs_load->type;
    int element_width = lhs_load_type.bytes();
    auto lhs_type = lhs_load_type.with_lanes(1024 / element_width);
    auto lhs = Call::make(lhs_type, "tile_load", {tile_x, tile_r * element_width, lhs_var, lhs_tile.base * element_width, lhs_tile.stride[0] * element_width}, Call::Intrinsic);

    auto rhs_var = Variable::make(Handle(), rhs_load->name);
    const auto &rhs_load_type = rhs_load->type;
    auto rhs_type = rhs_load_type.with_lanes(1024 / element_width);

    auto rhs = Call::make(rhs_type, "tile_load", {tile_r / (4 / element_width), tile_y * 4, rhs_var, rhs_base * element_width, rhs_stride}, Call::Intrinsic);
    auto res_type = amx_op_type_result_type(op_type);

    // {rows, colbytes, acc, out, lhs, rhs}
    auto out = Load::make(res_type, new_name, Ramp::make(0, 1, 256), {}, {}, const_true(256), {});

    // 4 bytes for i32, f32
    auto colbytes = tile_y * 4;
    auto matmul = Call::make(res_type, "tile_matmul", {tile_x, colbytes, tile_r, out, lhs, rhs}, Call::Intrinsic);
    auto store = Store::make(new_name, matmul, Ramp::make(0, 1, 256), Parameter(), const_true(256), ModulusRemainder());
    return {true, std::move(store), tile_x, tile_y, tile_r};
}

Stmt convert_to_zero(const Store *op, int tile_x, int tile_y, const string &new_name) {
    if (const auto *ramp = op->index.as<Ramp>()) {
        if (const auto *bcast = op->value.as<Broadcast>()) {
            if (is_const_one(ramp->stride) &&
                is_const_zero(bcast->value) &&
                (bcast->lanes == tile_x * tile_y)) {
                auto rows = Cast::make(Int(16), tile_x);
                auto bytes = op->value.type().bytes();
                auto colbytes = Cast::make(Int(16), tile_y * bytes);
                const auto &store_type = op->value.type();
                // will be f32 or i32
                auto tile_zero_type = store_type.with_lanes(1024 / store_type.bytes());
                auto val = Call::make(tile_zero_type, "tile_zero", {rows, colbytes}, Call::Intrinsic);
                auto store = Store::make(new_name, std::move(val), Ramp::make(0, 1, 256), Parameter(), const_true(256), ModulusRemainder());
                return store;
            }
        }
    }
    return {};
}

Stmt convert_to_tile_store(const Store *op, const string &amx_name, int tile_x, int tile_y) {
    auto tile = get_2d_tile_index(op->index);
    if (tile.result && tile.extent[0] == tile_x && tile.extent[1] == tile_y) {
        auto out = Variable::make(Handle(), op->name);
        auto tile_type = op->value.type().with_lanes(256);
        auto tile_val = Load::make(tile_type, amx_name, Ramp::make(0, 1, 256), {}, {}, const_true(256), {});
        auto bytes = op->value.type().bytes();
        internal_assert(bytes == 4) << "AMX store only supported for int32 and float32 output, not for " << op->value.type() << "\n";
        // {tile_x, tile_y, var, base, stride}
        auto store = Call::make(Int(32), "tile_store", {tile_x, tile_y * bytes, std::move(out), tile.base * bytes, tile.stride[0] * bytes, std::move(tile_val)}, Call::Intrinsic);
        return Evaluate::make(std::move(store));
    }
    return {};
}

class ExtractTileOperations : public IRMutator {
    using IRMutator::visit;

    string tile_name;
    string amx_name;
    vector<Stmt> pending_stores;
    bool in_allocate = false;
    int found_tile_x = -1;
    int found_tile_y = -1;
    int found_tile_r = -1;
    AMXOpType op_type;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::AMXTile) {
            user_assert(
                (op->type.is_int() && op->type.bits() == 32) ||
                (op->type.is_float() && op->type.bits() == 32))
                << "scheduled tile operations must yield 32-bit integers or 32-bit floats";

            if (op->type.is_int() && op->type.bits() == 32) {
                op_type = AMXOpType::Int8;
            } else {
                op_type = AMXOpType::Bfloat16;
            }

            user_assert(!in_allocate) << "Already in AMX allocation: " << amx_name;
            ScopedValue<string> old_amx_name(amx_name, op->name + ".amx");
            ScopedValue<string> old_tile_name(tile_name, op->name);
            ScopedValue<bool> old_in_alloc(in_allocate, true);
            Stmt body = op->body;

            pending_stores.clear();
            body = mutate(body);
            if (found_tile_x < 0 || found_tile_y < 0 || found_tile_r < 0) {
                return op;
            }
            if (!pending_stores.empty()) {
                // Really only need to go over the pending stores
                body = mutate(body);
            }

            auto tile_type = amx_op_type_result_type(op_type);
            return Allocate::make(amx_name, tile_type, MemoryType::AMXTile, {1}, const_true(), body);
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const Free *op) override {
        if (op->name != tile_name) {
            return op;
        }
        return Free::make(amx_name);
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (op->name != tile_name) {
            return IRMutator::visit(op);
        }

        auto body = mutate(op->body);
        return ProducerConsumer::make(amx_name, op->is_producer, std::move(body));
    }

    Expr visit(const Load *op) override {
        // Any tile load will be matched elsewhere, so a load here means that
        // the AMX tile is used outside of a tile instruction.
        user_assert(op->name != tile_name) << "AMX tile allocation used outside a tile instruction";
        return IRMutator::visit(op);
    }

    Stmt visit(const Store *op) override {
        if (op->name != tile_name) {
            const auto *load = op->value.as<Load>();
            if (!load || load->name != tile_name) {
                return op;
            }
            auto store = convert_to_tile_store(op, amx_name, found_tile_x, found_tile_y);
            user_assert(store.defined()) << "Store to AMX tile allocation of a non-tile value";
            return store;
        }

        auto matmul = convert_to_matmul(op, amx_name, op_type);
        if (matmul.result) {
            user_assert(
                (found_tile_x < 0 || matmul.tile_x == found_tile_x) &&
                (found_tile_y < 0 || matmul.tile_y == found_tile_y) &&
                (found_tile_r < 0 || matmul.tile_r == found_tile_r))
                << "Found different tile sizes for AMX tile allocation";
            found_tile_x = matmul.tile_x;
            found_tile_y = matmul.tile_y;
            found_tile_r = matmul.tile_r;

            return matmul.stmt;
        }

        if (found_tile_x < 0 || found_tile_y < 0) {
            pending_stores.emplace_back(op);
            return op;
        }

        auto zero = convert_to_zero(op, found_tile_x, found_tile_y, amx_name);
        if (zero.defined()) {
            return zero;
        }

        // Otherwise there is some other operation using the allocation, so we cannot use the AMX instructions
        user_error << "Found non-tile operations for AMX tile allocation";
        return op;
    }
};

}  // namespace

namespace EqSatExtensions {

Location from_memory_type(MemoryType memtype) {
    if (memtype == MemoryType::AMXTile) {
        return Location::AMX;
    } else if (memtype == MemoryType::WMMAAccumulator) {
        return Location::WMMA_C;
    } else if (memtype == MemoryType::WMMAA) {
        return Location::WMMA_A;
    } else if (memtype == MemoryType::WMMAB) {
        return Location::WMMA_B;
    } else {
        return Location::Mem;
    }
}

Expr GLoad::make(Type type, std::shared_ptr<Var> name, Expr index, Buffer<> image, Parameter param, Expr predicate, ModulusRemainder alignment) {
    internal_assert(predicate.defined()) << "Load with undefined predicate\n";
    internal_assert(index.defined()) << "Load of undefined\n";
    internal_assert(type.lanes() == index.type().lanes()) << "Vector lanes of Load must match vector lanes of index\n";
    internal_assert(type.lanes() == predicate.type().lanes())
        << "Vector lanes of Load must match vector lanes of predicate\n";

    GLoad *node = new GLoad();
    node->type = type;
    node->name = std::move(name);
    node->predicate = std::move(predicate);
    node->index = std::move(index);
    node->image = std::move(image);
    node->param = std::move(param);
    node->alignment = alignment;
    return node;
}

Expr GVariable::make(Type type, std::shared_ptr<Var> name) {
    GVariable *node = new GVariable;
    node->type = type;
    node->name = std::move(name);
    return node;
}

// Inserting LocToLoc nodes to represent the data movement between different memory types
// The only case we need to insert an AMXToMem node is when we are loading to an AMX tile
// The only case we need to insert a MemToAMX node is when we are storing to an AMX tile
class AnnotateDataMovement : public IRMutator {
    using IRMutator::visit;

public:
    AnnotateDataMovement() = default;
    bool no_accelerator_allocations = true;

protected:
    std::map<string, MemoryType> vars_memory_type;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::AMXTile) {
            user_assert(
                (op->type.is_int_or_uint() && op->type.bits() == 32) ||
                (op->type.is_float() && op->type.bits() == 32) ||
                (op->type.is_int_or_uint() && op->type.bits() == 8) ||
                (op->type.is_bfloat() && op->type.bits() == 16))
                << "scheduled tile operations must yield 32-bit integers or 32-bit floats for output, 8-bit integers or 16-bit floats for input";
            user_assert(vars_memory_type.find(op->name) == vars_memory_type.end()) << "Duplicate variable name: " << op->name;

        } else if (op->memory_type == MemoryType::WMMAAccumulator) {
            user_assert(op->type.is_float() && (op->type.bits() == 32 || op->type.bits() == 16)) << "currently only support float16xfloat16 to float32 for WMMA";

        } else if (op->memory_type != MemoryType::WMMAA && op->memory_type != MemoryType::WMMAB) {
            return IRMutator::visit(op);
        }

        no_accelerator_allocations = false;

        std::map<string, MemoryType> curr_vars_memory_type(this->vars_memory_type);
        curr_vars_memory_type[op->name] = op->memory_type;
        ScopedValue<std::map<string, MemoryType>> old_tile_vars(vars_memory_type, std::move(curr_vars_memory_type));

        Stmt body = mutate(op->body);

        return Allocate::make(op->name, op->type, op->memory_type, op->extents, const_true(), body);
    }

    Expr visit(const Load *load) override {
        auto result = vars_memory_type.find(load->name);
        if (result != vars_memory_type.end()) {
            internal_assert(is_const_one(load->predicate)) << "Only constant predicate is supported in accelerator tiles";
            return LocToLoc::make(load, load->type, from_memory_type(result->second), Location::Mem);
        } else {
            return load;
        }
    }

    Stmt visit(const Store *store) override {
        auto result = vars_memory_type.find(store->name);
        if (result != vars_memory_type.end()) {
            // `Load` can only occur in store->value,
            Expr value = LocToLoc::make(mutate(store->value), store->value.type(), Location::Mem, from_memory_type(result->second));
            internal_assert(is_const_one(store->predicate)) << "Only constant predicate is supported";
            return Store::make(
                store->name,
                value,
                store->index,
                store->param,
                store->predicate,
                store->alignment);
        } else {
            return IRMutator::visit(store);
        }
    }
};

std::string PLACEHOLDER_PREFIX = "collectstoresplaceholder";

struct CollectStores : public EqSatIRMutator {
    using EqSatIRMutator::visit;
    std::map<std::string, Stmt> stores;

    Stmt visit(const Store *op) override {
        auto no = stores.size();
        auto placeholder = PLACEHOLDER_PREFIX + op->name + ".eqsat." + std::to_string(no);
        placeholder = replace_all(placeholder, "$", "AAA");
        placeholder = replace_all(placeholder, ".", "BBB");
        stores[placeholder] = op;
        return Store::make(placeholder, op->value, op->index, op->param, op->predicate, op->alignment);
    }
};

class RemoveGLoadsAndGVars : public EqSatIRMutator {
    using EqSatIRMutator::visit;

public:
    struct PrologueStmt {
        std::string name;
        Expr expr;
        std::set<string> free_vars;
        std::deque<std::pair<IRNodeType, std::string>> trace;

        PrologueStmt(std::string name, Expr expr) {
            this->name = std::move(name);
            this->expr = std::move(expr);
            FreeVars free_vars;
            free_vars.mutate(this->expr);
            this->free_vars = free_vars.free_vars;
        }

        Type type() {
            const Call *call_stmt = this->expr.as<Call>();
            if (call_stmt && call_stmt->name.find("wmma.store.d") != string::npos) {
                int bits;
                if (call_stmt->name.substr(call_stmt->name.length() - 2) == "32") {
                    bits = 32;
                } else if (call_stmt->name.substr(call_stmt->name.length() - 2) == "16") {
                    bits = 16;
                } else {
                    internal_error << "Cannot tell the type of the tile to be stored";
                }
                return Float(bits, 256);
            } else {
                return this->expr.type();
            }
        }
    };

    explicit RemoveGLoadsAndGVars(const std::string &prefix)
        : prefix(prefix) {
    }

    const std::vector<PrologueStmt> &get_prologues() {
        return this->prologues;
    }

    class FreeVars : public EqSatIRMutator {
        using EqSatIRMutator::visit;

    public:
        std::set<string> free_vars;

    protected:
        Expr visit(const Variable *op) override {
            free_vars.insert(op->name);
            return EqSatIRMutator::visit(op);
        }
    };

protected:
    std::string prefix;
    int store_no = 0;
    std::vector<PrologueStmt> prologues;
    Expr visit(const GLoad *load) override {
        if (const StringVar *v = load->name->to_string_var()) {
            return Load::make(load->type, v->name, EqSatIRMutator::mutate(load->index), load->image, load->param, EqSatIRMutator::mutate(load->predicate), load->alignment);
        } else if (const ExprVar *v = load->name->to_expr_var()) {
            // We traverse children first to ensure that the prologues are in the correct order
            Expr predicate = EqSatIRMutator::mutate(load->predicate);
            Expr index = EqSatIRMutator::mutate(load->index);

            std::string name = prefix + std::to_string(store_no++);
            prologues.emplace_back(name, mutate(v->expr));
            return Load::make(load->type, name, index, load->image, load->param, predicate, load->alignment);
        } else {
            internal_error << "GLoad name must be a StringVar or ExprVar\n";
            return {};
        }
    }

    Expr visit(const GVariable *var) override {
        if (const StringVar *v = var->name->to_string_var()) {
            return Variable::make(var->type, v->name);
        } else if (const ExprVar *v = var->name->to_expr_var()) {
            auto name = prefix + std::to_string(store_no++);
            prologues.emplace_back(name, mutate(v->expr));
            return Variable::make(var->type, name);
        } else {
            internal_error << "GVariable name must be a StringVar\n";
            return {};
        }
    }
};

struct SubstKernelLoads : public EqSatIRMutator {
    using EqSatIRMutator::visit;

    SubstKernelLoads(std::string old_buffer_name, std::string new_buffer_name, Expr offset) {
        this->old_buffer_name = old_buffer_name;
        this->new_buffer_name = new_buffer_name;
        this->offset = offset;
    }

    Expr visit(const Call *call) override {
        // Check all loads
        if (starts_with(call->name, "wmma.load")) {
            const Variable *buff = call->args[0].as<Variable>();
            if (buff && buff->name == old_buffer_name) {
                std::vector<Expr> args = call->args;
                args[0] = Variable::make(args[0].type(), new_buffer_name);
                args[1] = args[1] + offset;
                return Call::make(call->type, call->name, args, call->call_type, call->func, call->value_index, call->image, call->param);
            }
        }
        return EqSatIRMutator::visit(call);
    }

    std::string old_buffer_name;
    std::string new_buffer_name;
    Expr offset;
};

struct ExtractWriteVars : public EqSatIRVisitor {
    using EqSatIRVisitor::visit;

    ExtractWriteVars(Stmt body) {
        body.accept(this);
    }

    void visit(const Store *op) override {
        write_vars.insert(op->name);
    }

    std::set<std::string> get_write_vars() {
        return write_vars;
    }

private:
    std::set<std::string> write_vars;
};

struct InsertPendingDefinition : public EqSatIRMutator {
    using EqSatIRMutator::visit;
    using PrologueStmt = RemoveGLoadsAndGVars::PrologueStmt;

    InsertPendingDefinition(Stmt program) {
        this->program = program;
    }

    bool attempt_insert(IRNodeType type, std::string name, PrologueStmt pending_definition) {
        this->name = name;
        this->type = type;
        this->_pd = pending_definition;
        this->inserted = false;
        program = mutate(program);
        return inserted;
    }

    Stmt get_program() {
        return program;
    }

    Stmt insert(Stmt body) {
        PrologueStmt pd = this->_pd.value();

        const int lanes = pd.type().lanes();
        Stmt store_stmt;

        const Call *call_stmt = pd.expr.as<Call>();
        if (call_stmt && call_stmt->name.find("wmma.store.d") != string::npos) {
            vector<Expr> args = call_stmt->args;
            args[0] = Variable::make(Handle(), pd.name);
            store_stmt = Evaluate::make(Call::make(call_stmt->type, call_stmt->name, args, call_stmt->call_type));
            body = Block::make(store_stmt, body);
            body = Allocate::make(pd.name, pd.type().with_lanes(1), MemoryType::Auto, {pd.type().lanes()}, const_true(pd.type().lanes()), body);
        } else if (call_stmt && call_stmt->name == "ConvolutionShuffle" && call_stmt->args.size() == 6 && false) {
            // yz: I disabled this
            Expr dist_expr = Call::make(
                call_stmt->type.with_lanes(lanes),
                "DistributedConvolutionShuffle",
                call_stmt->args,
                call_stmt->call_type,
                call_stmt->func,
                call_stmt->value_index,
                call_stmt->image,
                call_stmt->param);
            store_stmt = Store::make(pd.name, dist_expr, Ramp::make(0, 1, lanes), Parameter(), const_true(lanes), ModulusRemainder());
            // store_stmt = For::make("conv_shuffle.thread_id_x", 0, 32, ForType::GPULane, Partition::Auto, DeviceAPI::CUDA, store_stmt);

            body = Block::make(store_stmt, body);

            BufferBuilder builder;
            builder.host = Variable::make(Handle(), pd.name);
            builder.type = pd.type().with_lanes(1);
            builder.dimensions = 1;
            builder.mins.push_back(0);
            builder.extents.push_back(lanes);
            builder.strides.push_back(1);
            body = LetStmt::make(pd.name + ".buffer", builder.build(), body);
            body = Allocate::make(pd.name, pd.type().with_lanes(1), MemoryType::Auto, {pd.type().lanes()}, const_true(pd.type().lanes()), body);
        } else if (call_stmt && (call_stmt->name == "ConvolutionShuffle" || call_stmt->name == "ConvolutionShuffle+")) {
            store_stmt = Store::make(pd.name, pd.expr, Ramp::make(0, 1, lanes), Parameter(), const_true(lanes), ModulusRemainder());

            body = Block::make(store_stmt, body);

            BufferBuilder builder;
            builder.host = Variable::make(Handle(), pd.name);
            builder.type = pd.type().with_lanes(1);
            builder.dimensions = 1;
            builder.mins.push_back(0);
            builder.extents.push_back(lanes);
            builder.strides.push_back(1);
            body = LetStmt::make(pd.name + ".buffer", builder.build(), body);
            body = Allocate::make(pd.name, pd.type().with_lanes(1), MemoryType::Auto, {pd.type().lanes()}, const_true(pd.type().lanes()), body);
        } else {
            store_stmt = Store::make(pd.name, pd.expr, Ramp::make(0, 1, lanes), Parameter(), const_true(lanes), ModulusRemainder());
            body = Block::make(store_stmt, body);
            body = Allocate::make(pd.name, pd.type().with_lanes(1), MemoryType::Auto, {pd.type().lanes()}, const_true(pd.type().lanes()), body);
        }

        return body;
    }

    bool free_vars_are_modified(Stmt s) {
        ExtractWriteVars ewv(s);
        auto write_vars = ewv.get_write_vars();

        PrologueStmt pd = this->_pd.value();

        if (std::any_of(pd.free_vars.begin(), pd.free_vars.end(),
                        [&write_vars](const std::string &var) {
                            return write_vars.find(var) != write_vars.end();
                        })) {
            return true;
        } else {
            return false;
        }
    }

    Stmt visit(const For *op) override {
        if (op->node_type == type && op->name == name && !free_vars_are_modified(op->body)) {
            Stmt body = insert(op->body);
            this->inserted = true;
            return For::make(op->name, op->min, op->extent, op->for_type, op->partition_policy, op->device_api, body);
        } else {
            return EqSatIRMutator::visit(op);
        }
    }

    Stmt visit(const LetStmt *op) override {
        if (op->node_type == type && op->name == name && !free_vars_are_modified(op->body)) {
            Stmt body = insert(op->body);
            this->inserted = true;
            return LetStmt::make(op->name, op->value, body);
        } else {
            return EqSatIRMutator::visit(op);
        }
    }

    Stmt visit(const Allocate *op) override {
        if (op->node_type == type && op->name == name && !free_vars_are_modified(op->body)) {
            Stmt body = insert(op->body);
            this->inserted = true;
            return Allocate::make(op->name, op->type, op->memory_type, op->extents, op->condition, body);
        } else {
            return EqSatIRMutator::visit(op);
        }
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (op->node_type == type && op->name == name && !free_vars_are_modified(op->body)) {
            Stmt body = insert(op->body);
            this->inserted = true;
            return ProducerConsumer::make(op->name, op->is_producer, body);
        } else {
            return EqSatIRMutator::visit(op);
        }
    }

private:
    Stmt program;
    std::string name;
    IRNodeType type;
    std::optional<PrologueStmt> _pd;
    bool inserted;
};

struct SubstStores : public EqSatIRMutator {
    using EqSatIRMutator::visit;
    using PrologueStmt = RemoveGLoadsAndGVars::PrologueStmt;

    const std::map<std::string, Stmt> &stores;
    std::set<std::string> avail_vars;

    std::vector<PrologueStmt> pending_definitions;

    std::deque<std::pair<IRNodeType, std::string>> trace;

    Stmt program;

    SubstStores(Stmt program, std::map<std::string, Stmt> &&stores)
        : stores(stores), program(program) {
    }

    Stmt get_mutated_program() {
        Stmt out = this->mutate(program);

        // Delete duplicate pending definitions
        std::vector<PrologueStmt> unique_pending_definitions;
        for (size_t i = 0; i < pending_definitions.size(); i++) {
            bool unique = true;
            for (size_t j = 0; j < unique_pending_definitions.size(); j++) {
                // If they are exactly equal, we can replace the first with the second
                if (equal(pending_definitions[i].expr, unique_pending_definitions[j].expr)) {
                    debug(0) << "Found equals: " << pending_definitions[i].name << " and " << unique_pending_definitions[j].name << ": " << pending_definitions[i].expr << "\n";
                    SubstKernelLoads subst_kernel_loads(pending_definitions[i].name, unique_pending_definitions[j].name, 0);
                    out = subst_kernel_loads.mutate(out);
                    unique = false;
                    break;
                }
            }
            if (unique) {
                unique_pending_definitions.push_back(pending_definitions[i]);
            }
        }

        pending_definitions = unique_pending_definitions;

        // Merge convolution shuffles till convergence
        bool merge_happened = true;
        while (merge_happened) {
            merge_happened = false;

            std::vector<PrologueStmt> merged_pending_definitions;

            for (size_t i = 0; i < pending_definitions.size(); i++) {
                for (size_t j = 0; j < pending_definitions.size(); j++) {
                    if (i == j) continue;  // Skip self-comparison

                    // If they can be merged, merge them
                    const Call *c0 = pending_definitions[i].expr.as<Call>();
                    const Call *c1 = pending_definitions[j].expr.as<Call>();

                    if (c0 && c1 && c0->name == c1->name && c0->name.find("ConvolutionShuffle") != string::npos) {
                        bool can_merge_0_1 = can_merge_shuffles(c0, c1);
                        bool can_merge_1_0 = can_merge_shuffles(c1, c0);
                        if (can_merge_0_1 || can_merge_1_0) {
                            debug(0) << "Can merge " << pending_definitions[i].expr << " and " << pending_definitions[j].expr << "\n";
                            // Add all pending definitions to merged_pending_definitions, except the one we are merging
                            for (size_t k = 0; k < pending_definitions.size(); k++) {
                                if (k != i && k != j) {
                                    merged_pending_definitions.push_back(pending_definitions[k]);
                                }
                            }
                            // Add the merged definition
                            if (can_merge_0_1) {
                                pending_definitions[i].expr = merge_shuffles(c0, c1);
                                merged_pending_definitions.push_back(pending_definitions[i]);

                                // Update loads
                                int offset = c0->type.lanes();
                                SubstKernelLoads subst_kernel_loads(pending_definitions[j].name, pending_definitions[i].name, offset);
                                out = subst_kernel_loads.mutate(out);
                            } else {
                                pending_definitions[j].expr = merge_shuffles(c1, c0);
                                merged_pending_definitions.push_back(pending_definitions[j]);

                                // Update loads
                                int offset = c1->type.lanes();
                                SubstKernelLoads subst_kernel_loads(pending_definitions[i].name, pending_definitions[j].name, offset);
                                out = subst_kernel_loads.mutate(out);
                            }

                            merge_happened = true;
                            break;
                        }
                    }
                }
                if (merge_happened) break;
            }

            // If no merges happened, preserve the original definitions
            if (merge_happened) {
                pending_definitions = merged_pending_definitions;
            }
        };

        debug(0) << "Pending definitions:\n"
                 << pending_definitions.size() << "\n";
        for (auto &pending_definition : pending_definitions) {
            debug(0) << "Pending definition: " << pending_definition.name << " " << pending_definition.expr << "\n";
        }

        InsertPendingDefinition ipd(out);

        auto pd_it = pending_definitions.begin();
        while (pd_it != pending_definitions.end()) {
            auto &pending_definition = *pd_it;

            std::set<std::string> vars_in_scope;
            for (auto it = pending_definition.trace.rbegin(); it != pending_definition.trace.rend(); ++it) {
                IRNodeType t = it->first;
                std::string name = it->second;

                // Add variables introduced by the statement into scope
                vars_in_scope.insert(name);

                // If all needed variables are in scope, we can try to insert the pending definition.
                if (std::includes(vars_in_scope.begin(), vars_in_scope.end(), pending_definition.free_vars.begin(), pending_definition.free_vars.end())) {

                    // Before we insert, we need to check if the variable/buffer we are reading is modified down the line
                    bool inserted = ipd.attempt_insert(t, name, pending_definition);
                    if (inserted) {
                        pd_it = pending_definitions.erase(pd_it);
                        break;
                    }
                }
            }
        }

        return ipd.get_program();
    }

    bool can_merge_shuffles(const Call *new_call, const Call *existing_call) {
        // Check if they can be merged
        bool can_merge = true;

        // Matching buffer, stride and tile size
        can_merge = can_merge && equal(new_call->args[0], existing_call->args[0]);
        can_merge = can_merge && equal(new_call->args[2], existing_call->args[2]);
        can_merge = can_merge && equal(new_call->args[4], existing_call->args[4]);

        // Read contiguous chunks
        Expr base1 = existing_call->args[1];
        Expr stride = existing_call->args[2];
        Expr length1 = existing_call->args[3];
        Expr base2 = new_call->args[1];
        can_merge = can_merge && equal(simplify(base1 + length1 * stride), simplify(base2));

        return false;
        //        return can_merge;
    }

    Expr merge_shuffles(const Call *new_call, const Call *existing_call) {
        Expr base1 = existing_call->args[1];
        Expr stride = existing_call->args[2];
        Expr length1 = existing_call->args[3];

        Expr buffer = existing_call->args[0];
        Expr length2 = new_call->args[3];
        Expr taps = simplify(length1 + length2);
        Expr pixels = existing_call->args[4];

        Expr tiles1 = (existing_call->args.size() > 5) ? existing_call->args[5] : make_one(Int(32));
        Expr tiles2 = (new_call->args.size() > 5) ? new_call->args[5] : make_one(Int(32));
        Expr tile_cnt = tiles1 + tiles2;

        // Create the merged ConvolutionShuffle call
        std::vector<Expr> merged_args = {buffer, base1, stride, taps, pixels, tile_cnt};
        Expr merged_call = Call::make(
            existing_call->type.with_lanes(existing_call->type.lanes() + new_call->type.lanes()),
            "ConvolutionShuffle",
            merged_args,
            existing_call->call_type,
            existing_call->func,
            existing_call->value_index,
            existing_call->image,
            existing_call->param);

        return simplify(merged_call);
    }

    auto merge_convolution_shuffles(std::vector<RemoveGLoadsAndGVars::PrologueStmt> prologues, Stmt &s) {
        // If the prologue is a ConvolutionShuffle call, we can merge it with an existing pending_definition
        // if it reads from contiguous memory locations
        std::vector<RemoveGLoadsAndGVars::PrologueStmt> rem_prologues;
        for (const auto &prologue : prologues) {
            bool merged = false;

            // If new prologue is a ConvolutionShuffle
            const Call *new_call = prologue.expr.as<Call>();
            if (new_call && new_call->name == "ConvolutionShuffle") {
                // And if a pending_definition is also a ConvolutionShuffle
                for (size_t idx = 0; idx < pending_definitions.size(); ++idx) {
                    auto &pending_definition = pending_definitions[idx];

                    // It may be exactly equal to a pending definition
                    if (equal(pending_definition.expr, prologue.expr)) {
                        debug(0) << "Found equals: " << pending_definition.expr << " and " << prologue.expr << "\n";
                        debug(0) << "Replacing " << prologue.name << " with " << pending_definition.name << "\n";
                        SubstKernelLoads subst_kernel_loads(prologue.name, pending_definition.name, 0);
                        s = subst_kernel_loads.mutate(s);
                        merged = true;
                        break;
                    }

                    // Or it may be contiguous with an existing definition
                    const Call *existing_call = pending_definition.expr.as<Call>();
                    if (existing_call && existing_call->name == "ConvolutionShuffle") {
                        if (can_merge_shuffles(new_call, existing_call)) {
                            pending_definition.expr = merge_shuffles(new_call, existing_call);
                            debug(0) << "Merge result: " << pending_definition.expr << "\n";

                            int offset = existing_call->type.lanes();
                            SubstKernelLoads subst_kernel_loads(prologue.name, pending_definition.name, offset);
                            s = subst_kernel_loads.mutate(s);
                            merged = true;
                            break;
                        }
                    }
                }
            }
            if (!merged) {
                rem_prologues.push_back(prologue);
            }
        }

        return rem_prologues;
    }

    Stmt visit(const For *op) override {
        trace.push_front(std::make_pair(IRNodeType::For, op->name));

        const string name = op->name;
        avail_vars.insert(name);
        Stmt body = mutate(op->body);
        avail_vars.erase(name);

        trace.pop_front();

        return For::make(op->name, op->min, op->extent, op->for_type, op->partition_policy, op->device_api, body);
    }

    Stmt visit(const LetStmt *op) override {
        trace.push_front(std::make_pair(IRNodeType::LetStmt, op->name));

        string name = op->name;
        avail_vars.insert(name);
        Stmt body = mutate(op->body);
        avail_vars.erase(name);

        trace.pop_front();

        return LetStmt::make(op->name, mutate(op->value), body);
    }

    Stmt visit(const Allocate *op) override {
        trace.push_front(std::make_pair(IRNodeType::Allocate, op->name));

        string name = op->name;
        avail_vars.insert(name);
        Stmt body = mutate(op->body);
        avail_vars.erase(name);

        trace.pop_front();

        return Allocate::make(op->name, op->type, op->memory_type, op->extents, op->condition, body, op->new_expr, op->free_function, op->padding);
    }

    Stmt visit(const ProducerConsumer *op) override {
        trace.push_front(std::make_pair(IRNodeType::ProducerConsumer, op->name));
        Stmt body = mutate(op->body);
        trace.pop_front();
        return ProducerConsumer::make(op->name, op->is_producer, body);
    }

    Stmt visit(const Store *op) override {
        internal_assert(op->name.find(PLACEHOLDER_PREFIX) == 0)
            << "All stores should have been replaced with a place holder";
        auto it = stores.find(op->name);
        internal_assert(it != stores.end()) << "Store not found";

        Stmt s = it->second;
        auto prefix = replace_all(op->name.substr(PLACEHOLDER_PREFIX.length()), "AAA", "$");
        prefix = replace_all(prefix, "BBB", ".");
        RemoveGLoadsAndGVars remover(prefix);
        s = remover.mutate(s);

        auto prologues = remover.get_prologues();
        // prologues = merge_convolution_shuffles(prologues, s);

        for (auto &prologue : prologues) {
            prologue.trace = trace;
        }

        pending_definitions.insert(pending_definitions.end(), prologues.begin(), prologues.end());

        return s;
    }
};

Type convert_to_tile_type(Type type) {
    Type tile_type;
    if (type.is_bfloat() && type.bits() == 16) {
        tile_type = BFloat(16, 512);
    } else if (type.is_int() || type.is_uint() || type.is_float()) {  // is_float() returns true for bfloat
        tile_type = type.with_lanes(1024 / type.bytes());
    } else {
        internal_error << "Unsupported type for AMX tile";
    }
    return tile_type;
}

// The expression returned by EqSat, is not necessarily compatiable with the
// runtime, since the runtime always views an AMX Tile as a 256-lane vector.
class EnforceAMXShape : public IRMutator {
    using IRMutator::visit;

protected:
    std::vector<string> tile_vars;
    std::unordered_map<string, int> amx_tile_size;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::AMXTile) {
            std::vector<string> curr_tile_vars(this->tile_vars);
            curr_tile_vars.push_back(op->name);
            ScopedValue<vector<string>> old_tile_vars(tile_vars, std::move(curr_tile_vars));
            auto body = mutate(op->body);
            internal_assert(amx_tile_size.count(op->name)) << "AMX tile size not found";

            auto tile_type = convert_to_tile_type(op->type);
            return Allocate::make(op->name, tile_type, MemoryType::AMXTile, {amx_tile_size[op->name]}, const_true(), body);
        } else {
            return IRMutator::visit(op);
        }
    }

    int get_nth_tile_from_tile_index_amx(const Expr &e) {
        const Ramp *ramp = e.as<Ramp>();
        if (!ramp) {
            internal_error << "AMX tile can only be indexed with ramp";
            return -1;
        }
        int vec_length = ramp->lanes;
        const auto basep = as_const_int(ramp->base);
        if (!basep.has_value()) {
            internal_error << "Only constant base is supported in AMX";
            return -1;
        }
        int base = basep.value();
        internal_assert(base % vec_length == 0) << "Cannot determine which AMX tile to load from";
        return base / vec_length;
    }

    Expr visit(const Load *load) override {
        if (std::find(tile_vars.begin(), tile_vars.end(), load->name) != tile_vars.end()) {
            int tile = get_nth_tile_from_tile_index_amx(load->index);
            amx_tile_size[load->name] = std::max(amx_tile_size[load->name], tile + 1);
            Type tile_type = convert_to_tile_type(load->type);
            int tile_lanes = tile_type.lanes();
            return Load::make(tile_type,
                              load->name,
                              Ramp::make(tile_lanes * tile, 1, tile_lanes),
                              load->image,
                              load->param,
                              const_true(tile_lanes),
                              load->alignment);
        } else {
            return load;
        }
    }

    Stmt visit(const Store *store) override {
        if (std::find(tile_vars.begin(), tile_vars.end(), store->name) != tile_vars.end()) {
            internal_assert(is_const_one(store->predicate)) << "Only constant predicate is supported";
            int tile = get_nth_tile_from_tile_index_amx(store->index);
            amx_tile_size[store->name] = std::max(amx_tile_size[store->name], tile + 1);
            int tile_lanes = convert_to_tile_type(store->value.type()).lanes();

            // There should not be a Load in places other than value,
            // so we don't need to mutate them.
            Expr value = mutate(store->value);
            return Store::make(
                store->name,
                value,
                Ramp::make(tile * tile_lanes, 1, tile_lanes),
                store->param,
                const_true(tile_lanes),
                store->alignment);
        } else {
            return IRMutator::visit(store);
        }
    }

    Expr visit(const Call *call) override {
        // Not included:
        // - tile_store since it returns a single value
        vector<string> intrinsic_names = {"tile_load", "tile_matmul", "tile_zero"};

        if (std::find(intrinsic_names.begin(), intrinsic_names.end(), call->name) != intrinsic_names.end()) {
            return Call::make(
                convert_to_tile_type(call->type),
                call->name,
                mutate(call->args),
                call->call_type,
                call->func,
                call->value_index,
                call->image,
                call->param);
        } else {
            return IRMutator::visit(call);
        }
    }
};

// The intrinsics for WMMA operations are internally represented as a single-thread operation.
// However, it should be a warp-level instruction. This pass wraps the WMMA operation with GPU_lanes
// and updates the types.
// More concretely, this pass
// (1) removes the gpu_thread(idx, 0, 1) loop and replaces it with a gpu_lane(idx, 0, 32) loop
// (2) shrinks the type of the WMMA intrinsics to be per lane
// This pass shares many code with EnforceAMXShape, but for now we keep them separate.
class EnforceWMMALanes : public IRMutator {
    using IRMutator::visit;

    std::map<string, MemoryType> tile_vars;
    std::map<string, Type> intrinsic_types = {
        {"wmma.load.a.sync.aligned.row.m16n16k16.f16", Int(32, 8)},
        {"wmma.load.b.sync.aligned.row.m16n16k16.f16", Int(32, 8)},
        {"wmma.load.b.sync.aligned.col.m16n16k16.f16", Int(32, 8)},
        {"wmma.mma.sync.aligned.row.row.m16n16k16.f32.f32", Float(32, 8)},
        {"wmma.mma.sync.aligned.row.col.m16n16k16.f32.f32", Float(32, 8)},
        {"wmma.load.c.sync.aligned.row.m16n16k16.f32", Float(32, 8)},
        {"wmma.mma.sync.aligned.row.row.m16n16k16.f16.f16", Float(32, 4)},
        {"wmma.mma.sync.aligned.row.col.m16n16k16.f16.f16", Float(32, 4)},

        {"wmma.load.a.sync.aligned.row.m32n8k16.f16", Int(32, 8)},
        {"wmma.load.b.sync.aligned.row.m32n8k16.f16", Int(32, 8)},
        {"wmma.mma.sync.aligned.row.row.m32n8k16.f32.f32", Float(32, 8)},
        {"wmma.load.c.sync.aligned.row.m32n8k16.f32", Float(32, 8)},
        {"wmma.load.c.sync.aligned.row.m32n8k16.f16", Float(32, 4)},
        {"wmma.mma.sync.aligned.row.row.m32n8k16.f16.f16", Float(32, 4)},

        {"wmma.load.a.sync.aligned.row.m8n32k16.f16", Int(32, 8)},
        {"wmma.load.b.sync.aligned.row.m8n32k16.f16", Int(32, 8)},
        {"wmma.mma.sync.aligned.row.row.m8n32k16.f32.f32", Float(32, 8)},
        {"wmma.load.c.sync.aligned.row.m8n32k16.f32", Float(32, 8)},
        {"wmma.load.c.sync.aligned.row.m8n32k16.f16", Float(32, 4)},
        {"wmma.mma.sync.aligned.row.row.m8n32k16.f16.f16", Float(32, 4)},
    };

    Expr get_nth_tile_from_tile_index_wmma(const Expr &e) {
        const Ramp *ramp = e.as<Ramp>();
        if (!ramp) {
            internal_error << "WMMA tile can only be indexed with ramp";
            return -1;
        }
        if (ramp->base.type().is_scalar()) {
            int vec_length = ramp->lanes;
            internal_assert(can_prove(ramp->base % vec_length == 0)) << "Cannot determine which WMMA tile to load from: " << ramp->base;
            return ramp->base / vec_length;
        } else if (const Ramp *inner_ramp = ramp->base.as<Ramp>()) {
            // It's a 2D tiling
            int inner_lanes = inner_ramp->lanes;
            int outer_lanes = ramp->lanes;
            auto inner_stride = as_const_int(inner_ramp->stride);
            auto outer_stride = as_const_int(ramp->stride);
            internal_assert(inner_stride && *inner_stride == 1);
            internal_assert(outer_stride);
            Expr base = inner_ramp->base;
            base /= inner_lanes;
            int f = *outer_stride / inner_lanes;
            return (base % f) + f * (base / (f * outer_lanes));
            // TODO: Asserts on divisibility
            // TODO: This assumes all accesses to this storage are tiled in the same way!
        } else {
            internal_error << "Cannot determine which WMMA tile to load from: " << e << "\n";
            return Expr{};
        }
    }

protected:
    bool wmma_used = false;

    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::WMMAAccumulator || op->memory_type == MemoryType::WMMAB || op->memory_type == MemoryType::WMMAA) {
            internal_assert(op->constant_allocation_size() != 0 && op->constant_allocation_size() % 32 == 0);
            internal_assert(op->type.lanes() == 1);

            std::map<string, MemoryType> curr_tile_vars(this->tile_vars);
            curr_tile_vars[op->name] = op->memory_type;
            ScopedValue<std::map<string, MemoryType>> old_tile_vars(tile_vars, std::move(curr_tile_vars));

            auto body = mutate(op->body);
            auto lanes = op->constant_allocation_size() / 32;
            if (op->type.bits() == 16) {
                lanes /= 2;
            }
            Type type = op->memory_type == MemoryType::WMMAAccumulator ? Float(32) : Int(32);
            return Allocate::make(op->name,
                                  type,
                                  op->memory_type, {lanes}, const_true(),
                                  body);
        } else {
            return IRMutator::visit(op);
        }
    }

    Expr visit(const Call *op) override {
        // if (
        //     op->name.find("load.a") != string::npos
        //     ||
        //     op->name.find("load.b") != string::npos
        // ) {
        //     wmma_used = true;
        //     return make_zero(Int(32, 8));

        // } else
        // if (op->name.find("row.row") != string::npos) {
        //     wmma_used = true;
        //     return make_one(Float(32, 8));

        // } else
        if (intrinsic_types.count(op->name)) {
            wmma_used = true;
            internal_assert(op->type.lanes() % 32 == 0);
            return Call::make(
                intrinsic_types[op->name],
                op->name,
                mutate(op->args),
                op->call_type,
                op->func,
                op->value_index,
                op->image,
                op->param);
        } else if (ends_with(op->name, ".ZERO") && starts_with(op->name, "wmma.load.c.sync.aligned.")) {
            wmma_used = true;
            return op->name.find(".f32.") != string::npos ? make_zero(Float(32, 8)) : make_zero(Float(32, 4));
        } else {
            if (starts_with(op->name, "wmma.store.d.sync.aligned.")) {
                wmma_used = true;
            }
            return IRMutator::visit(op);
        }
    }

    Expr visit(const Load *load) override {
        auto it = tile_vars.find(load->name);
        if (it != tile_vars.end()) {
            internal_assert(load->type.lanes() % 32 == 0);
            auto nth = get_nth_tile_from_tile_index_wmma(load->index);
            int tile_lanes = load->type.lanes() / 32 / (load->type.bits() == 32 ? 1 : 2);
            MemoryType mt = it->second;
            Type tile_type = mt == MemoryType::WMMAAccumulator ? Float(32, tile_lanes) : Int(32, tile_lanes);
            // If the accumulator is float 16, each tile takes 4 lanes instead of 8
            int tile_multiple = load->type.bits() == 32 ? 8 : 4;
            return Load::make(tile_type,
                              load->name,
                              Ramp::make(nth * tile_multiple, 1, tile_lanes),
                              load->image,
                              load->param,
                              const_true(tile_lanes),
                              load->alignment);
        } else {
            return load;
        }
    }

    Stmt visit(const Evaluate *op) override {
        internal_assert(!wmma_used);
        auto new_op = IRMutator::visit(op);
        if (wmma_used) {
            wmma_used = false;
            return For::make("wmma_gpu_lane.thread_id_x", 0, 32, ForType::GPULane, Partition::Auto, DeviceAPI::CUDA, new_op);
        } else {
            return new_op;
        }
    }

    Stmt visit(const Store *store) override {
        if (tile_vars.find(store->name) != tile_vars.end()) {
            internal_assert(is_const_one(store->predicate)) << "Only constant predicate is supported";
            internal_assert(store->value.type().lanes() % 32 == 0);
            int tile_multiple = store->value.type().bits() == 32 ? 8 : 4;
            int tile_lanes = store->value.type().lanes() / 32 / (store->value.type().bits() == 32 ? 1 : 2);
            auto nth = get_nth_tile_from_tile_index_wmma(store->index);

            // There should not be a Load in places other than value,
            // so we don't need to mutate them.
            internal_assert(!wmma_used);
            Expr value = mutate(store->value);
            internal_assert(wmma_used);
            wmma_used = false;

            auto op = Store::make(
                store->name,
                value,
                Ramp::make(nth * tile_multiple, 1, tile_lanes),
                store->param,
                const_true(tile_lanes),
                store->alignment);
            op = For::make("wmma_gpu_lane.thread_id_x", 0, 32, ForType::GPULane, Partition::Auto, DeviceAPI::CUDA, op);
            return op;
        } else {
            return IRMutator::visit(store);
        }
    }
};

class DesugarIntrinsics : public IRMutator {
    using IRMutator::visit;

protected:
    Stmt visit(const Store *store) override {
        const Call *rhs = store->value.as<Call>();
        if (rhs && on_gpu && rhs->name == "ConvolutionShuffle") {
            const auto &args = rhs->args;
            const auto *var = args[0].as<Variable>();
            auto base_r = args[1];
            auto stride_r = args[2];
            const auto l1_opt = as_const_int(args[3]);
            const auto l2_opt = as_const_int(args[4]);
            const auto steps_opt = as_const_int(args[5]);
            const auto offset_opt = as_const_int(args[6]);

            internal_assert(l1_opt && l2_opt && steps_opt && offset_opt);
            int l1 = (int)(*l1_opt);
            int l2 = (int)(*l2_opt);
            int steps = (int)(*steps_opt);
            int offset = (int)(*offset_opt);

            Expr j = Variable::make(Int(32), "lane.thread_id_x");
            Expr idx_into_filter = j + offset - Ramp::make(0, 1, l2) * steps;
            Expr clamped_idx = clamp(idx_into_filter,
                                     make_const(idx_into_filter.type(), 0),
                                     make_const(idx_into_filter.type(), l1 - 1));
            Expr mask = (clamped_idx == idx_into_filter);
            Expr v = Load::make(Float(16, l2), var->name, base_r + stride_r * clamped_idx, {}, {}, const_true(l2), {});
            v = select(mask, v, make_zero(Float(16, l2)));
            // Expr v = Load::make(Float(16, l2), var->name, base_r + stride_r * idx_into_filter, {}, {}, mask, {});

            const Ramp *r = store->index.as<Ramp>();
            internal_assert(r);
            Stmt s = Store::make(store->name, simplify(v),
                                 simplify(Ramp::make(r->base + j * r->stride * l2, r->stride, l2)),
                                 store->param, const_true(l2), {});
            s = For::make(j.as<Variable>()->name, 0, l1 / steps + l2, ForType::GPULane, Partition::Auto, DeviceAPI::CUDA, s);
            return s;
        } else if (rhs && on_gpu && rhs->name == "ConvolutionShuffle+") {
            const std::vector<Expr> &args = rhs->args;
            internal_assert(args.size() == 9);
            const auto *var = args[0].as<Variable>();
            auto base_r = args[1];
            auto stride_r = args[2];
            const auto l1_opt = as_const_int(args[3]);
            const auto l2_opt = as_const_int(args[4]);
            const auto steps_opt = as_const_int(args[5]);
            const auto offset_opt = as_const_int(args[6]);
            const auto repeat_stride_opt = as_const_int(args[7]);
            const auto repeat_count_opt = as_const_int(args[8]);
            if (!(var && l1_opt.has_value() && l2_opt.has_value() && offset_opt.has_value() && steps_opt.has_value() &&
                  repeat_stride_opt.has_value() && repeat_count_opt.has_value())) {
                internal_error << "ConvolutionShuffle+: arguments have unexpected type\n";
            }
            auto l1 = (int)l1_opt.value();
            auto l2 = (int)l2_opt.value();
            auto offset = (int)offset_opt.value();
            auto steps = (int)steps_opt.value();
            auto repeat_stride = (int)repeat_stride_opt.value();
            auto repeat_count = (int)repeat_count_opt.value();

            Expr j = Variable::make(Int(32), "lane.thread_id_x");
            std::vector<Stmt> stmts;
            for (int i = 0; i < l2; i++) {
                Expr k = Ramp::make(0, 1, repeat_count);
                Expr idx_into_filter = j + offset - i * steps;
                Expr clamped_idx = clamp(idx_into_filter,
                                         make_const(idx_into_filter.type(), 0),
                                         make_const(idx_into_filter.type(), l1 - 1));
                Expr mask = (clamped_idx == idx_into_filter);
                clamped_idx = clamped_idx + k * repeat_stride;

                Expr v = Load::make(Float(16, repeat_count), var->name, base_r + stride_r * clamped_idx, {}, {}, const_true(repeat_count), {});

                const Ramp *r = store->index.as<Ramp>();
                internal_assert(r);
                Expr store_idx =
                    Ramp::make((j * l2 + i) * repeat_count, 1, repeat_count);
                store_idx = r->base + store_idx * r->stride;
                Stmt s = Store::make(store->name, simplify(v), simplify(store_idx),
                                     store->param, const_true(repeat_count), {});
                stmts.push_back(s);
            }

            return For::make(j.as<Variable>()->name, 0, l1 / steps + l2, ForType::GPULane, Partition::Auto, DeviceAPI::CUDA, Block::make(stmts));
        }

        return IRMutator::visit(store);
    }

    bool on_gpu = false;
    Stmt visit(const For *op) override {
        ScopedValue<bool> old_on_gpu(on_gpu, on_gpu || (op->for_type == ForType::GPUBlock));
        return IRMutator::visit(op);
    }

    Expr visit(const Call *call) override {
        if (call->name == "KWayInterleave") {
            std::vector<Expr> args = call->args;
            internal_assert(args.size() == 3);
            auto pk = as_const_int(args[0]);
            Expr vec = mutate(args[1]);
            auto planes = as_const_int(args[2]);
            if (!pk.has_value() || !planes.has_value()) {
                internal_error << "KWayInterleave requires constant k and lanes\n";
                return {};
            }
            int K = pk.value();
            int lanes = planes.value();
            internal_assert(vec.type().lanes() % lanes == 0) << "Vector size must be a multiple of #lanes\n";
            internal_assert(lanes % K == 0) << "Number of lanes must be a multiple of k\n";
            int length = vec.type().lanes() / lanes;

            std::vector<int> indices;
            for (int i = 0; i < lanes; i += K) {
                for (int j = 0; j < length; j++) {
                    for (int k = 0; k < K; k++) {
                        indices.push_back((i + k) * length + j);
                    }
                }
            }
            return Shuffle::make({vec}, indices);

        } else if (call->name == "ConvolutionShuffle") {
            const std::vector<Expr> &args = call->args;
            internal_assert(args.size() == 7);
            const auto *var = args[0].as<Variable>();
            auto base_r = args[1];
            auto stride_r = args[2];
            const auto l1_opt = as_const_int(args[3]);
            const auto l2_opt = as_const_int(args[4]);
            const auto steps_opt = as_const_int(args[5]);
            const auto offset_opt = as_const_int(args[6]);
            if (!(var && l1_opt.has_value() && l2_opt.has_value() && offset_opt.has_value() && steps_opt.has_value())) {
                internal_error << "ConvolutionShuffle: arguments have unexpected type\n";
                return Expr();
            }
            auto l1 = l1_opt.value();
            auto l2 = l2_opt.value();
            auto offset = offset_opt.value();
            auto steps = steps_opt.value();
            auto ty = Float(16, l1);
            Expr vec1 = Load::make(ty, var->name, Ramp::make(base_r, stride_r, l1), {}, {}, const_true(l1), {});
            Expr vec2 = FloatImm::make(Float(16), 0);
            vector<int> indices;
            for (int j = 0; j < (l1 / steps) + l2; j++) {
                for (int i = 0; i < l2; i++) {
                    int idx_into_filter = j + offset - i * steps;
                    if (0 <= idx_into_filter && idx_into_filter < l1) {
                        indices.push_back(idx_into_filter);
                    } else {
                        indices.push_back(l1);
                    }
                }
            }
            auto v = Shuffle::make({vec1, vec2}, indices);
            return v;
        } else if (call->name == "ConvolutionShuffle+") {
            const std::vector<Expr> &args = call->args;
            internal_assert(args.size() == 9);
            const auto *var = args[0].as<Variable>();
            auto base_r = args[1];
            auto stride_r = args[2];
            const auto l1_opt = as_const_int(args[3]);
            const auto l2_opt = as_const_int(args[4]);
            const auto steps_opt = as_const_int(args[5]);
            const auto offset_opt = as_const_int(args[6]);
            const auto repeat_stride_opt = as_const_int(args[7]);
            const auto repeat_count_opt = as_const_int(args[8]);
            if (!(var && l1_opt.has_value() && l2_opt.has_value() && offset_opt.has_value() && steps_opt.has_value() &&
                  repeat_stride_opt.has_value() && repeat_count_opt.has_value())) {
                internal_error << "ConvolutionShuffle+: arguments have unexpected type\n";
                return Expr();
            }
            auto l1 = l1_opt.value();
            auto l2 = l2_opt.value();
            auto offset = offset_opt.value();
            auto steps = steps_opt.value();
            auto repeat_stride = repeat_stride_opt.value();
            auto repeat_count = repeat_count_opt.value();
            auto ty = Float(16, l1 * repeat_count);
            Expr vec1 = Load::make(ty, var->name,
                                   Ramp::make(
                                       Ramp::make(base_r, stride_r, l1),
                                       Broadcast::make(IntImm::make(Int(32), repeat_stride), l1),
                                       repeat_count),
                                   {}, {}, const_true(l1 * repeat_count), {});
            Expr vec2 = FloatImm::make(Float(16), 0);
            vector<int> indices;
            for (int j = 0; j < (l1 / steps) + l2; j++) {
                for (int i = 0; i < l2; i++) {
                    for (int k = 0; k < repeat_count; k++) {
                        int idx_into_filter = j + offset - i * steps;
                        if (0 <= idx_into_filter && idx_into_filter < l1) {
                            indices.push_back(idx_into_filter + k * l1);
                        } else {
                            indices.push_back(l1 * repeat_count);
                        }
                    }
                }
            }

            auto v = Shuffle::make({vec1, vec2}, indices);
            return v;
        } else {
            return IRMutator::visit(call);
        }
    }
};

// This class expects the following forms
//  Allocate b[...] from WMMAAccumulator
//    ...
//    gpu_lanes / gpu_threads
// and translates it to
//  ...
//  gpu_lanes / gpu_threads
//   Allocate b[...] from WMMAAccumulator
class PushWMMAAllocation : public IRMutator {
    using IRMutator::visit;

    vector<const Allocate *> alloc_ops;

protected:
    Stmt visit(const Allocate *op) override {
        if (op->memory_type == MemoryType::WMMAAccumulator || op->memory_type == MemoryType::WMMAA || op->memory_type == MemoryType::WMMAB) {
            ScopedValue<vector<const Allocate *>> old_alloc_ops(alloc_ops);
            alloc_ops.push_back(op);
            return mutate(op->body);
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const For *op) override {
        if (op->for_type == ForType::GPULane && !alloc_ops.empty()) {
            auto body = op->body;
            for (auto iter = alloc_ops.rbegin(); iter != alloc_ops.rend(); iter++) {
                const auto *alloc_op = *iter;
                auto type = alloc_op->type.with_lanes(1);
                auto extents = alloc_op->extents;
                extents.insert(extents.begin(), alloc_op->type.lanes());
                body = Allocate::make(alloc_op->name, type, alloc_op->memory_type, extents, alloc_op->condition, body);
            }
            return For::make(op->name, op->min, op->extent, op->for_type, op->partition_policy, op->device_api, body);
        } else {
            return For::make(op->name, op->min, op->extent, op->for_type, op->partition_policy, op->device_api, mutate(op->body));
        }
    }
};

}  // namespace EqSatExtensions

Stmt extract_tile_operations(const Stmt &s) {
    return ExtractTileOperations().mutate(s);
}

Stmt eqsat_extract_tile_operations(const Stmt &s) {
    auto annotate = EqSatExtensions::AnnotateDataMovement();
    auto annotated_s = annotate.mutate(s);
    if (annotate.no_accelerator_allocations) {
        return s;
    }
    EqSatExtensions::CollectStores collect_stores;
    auto result = collect_stores.mutate(annotated_s);
    auto &stores = collect_stores.stores;

    // collect bindings and run egglog
    std::vector<std::pair<std::string, std::string>> bindings;
    for (const auto &[name, op] : stores) {
        std::ostringstream oss;
        EqSatIRPrinter sprinter(oss);
        sprinter.print(op);
        bindings.emplace_back(name, oss.str());
    }
    auto output = run_egglog(std::move(bindings));
    auto optimized_programs = split_string(output, "\n");

    // collect output of egglog
    bool amx_synthesized = false;
    std::map<std::string, Stmt> new_stores;
    for (size_t i = 0; i < optimized_programs.size(); ++i) {
        if (optimized_programs[i].empty()) {
            continue;
        }
        auto &optimized = optimized_programs[i];
        auto &name = bindings[i].first;
        amx_synthesized = amx_synthesized || optimized.find("tile_matmul") != string::npos;

        EqSatExtensions::EqSatIRParser parser(optimized);

        new_stores[name] = parser.parse_stmt();
    }
    if (!amx_synthesized) {
        std::cerr << "amx NOT synthesized\n";
    } else {
        std::cerr << "amx synthesized\n";
    }

    // post-processing
    EqSatExtensions::SubstStores subst_stores(result, std::move(new_stores));
    result = subst_stores.get_mutated_program();
    for (auto pending_definition : subst_stores.pending_definitions) {
        std::cerr << "pending_definition: " << pending_definition.expr << std::endl;
    }
    internal_assert(subst_stores.pending_definitions.empty());
    result = EqSatExtensions::EnforceAMXShape().mutate(result);
    result = EqSatExtensions::EnforceWMMALanes().mutate(result);
    debug(0) << result << "\n";
    result = EqSatExtensions::DesugarIntrinsics().mutate(result);
    return result;
}

Stmt post_process_wmma(const Stmt &s) {
    return EqSatExtensions::PushWMMAAllocation().mutate(s);
}

std::string run_egglog(std::vector<std::pair<std::string, std::string>> &&binding) {
#include "egglog/main.tmpl.h"

    std::string egglog_prog = EGGLOG_PROG(std::move(binding));

    std::string filename = "/tmp/egglog_prog_" + std::to_string(getpid()) + ".egg";
    // Write the program to a file
    std::cout << "Writing egglog program to " << filename << std::endl;
    std::ofstream file(filename);
    file << egglog_prog << std::flush;

    int pipe_stdin[2];
    int pipe_stdout[2];

    if (pipe(pipe_stdin) < 0 || pipe(pipe_stdout) < 0) {
        internal_error << "Failed to create pipe for egglog";
        return "";
    }

    pid_t pid = fork();

    if (pid < 0) {
        internal_error << "Failed to exec egglog";
    }

    if (pid == 0) {
        const char *argv[] = {"egglog-halide-sidecar", nullptr};
        close(pipe_stdin[1]);
        close(pipe_stdout[0]);

        // Redirect stdin and stdout
        dup2(pipe_stdin[0], STDIN_FILENO);
        dup2(pipe_stdout[1], STDOUT_FILENO);
        int devnull = open("/dev/null", O_WRONLY | O_CREAT, 0666);
        dup2(devnull, STDERR_FILENO);
        close(pipe_stdin[0]);
        close(pipe_stdout[1]);
        execvp(argv[0], const_cast<char **>(argv));
        internal_error << "egglog failed to exec";
        return "";
    }

    close(pipe_stdout[1]);

    // Write to the subprocess's stdin
    size_t written = write(pipe_stdin[1], egglog_prog.c_str(), egglog_prog.size());
    internal_assert(written > 0);
    close(pipe_stdin[1]);

    // Read from the subprocess's stdout
    std::ostringstream oss;
    char buffer[128];
    ssize_t count;
    while ((count = read(pipe_stdout[0], buffer, sizeof(buffer) - 1)) > 0) {
        buffer[count] = '\0';  // Null-terminate the string
        oss << buffer;
    }

    close(pipe_stdout[0]);
    return oss.str();
}

template<>
void ExprNode<EqSatExtensions::LocToLoc>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "LocToLoc can only be visited by EqSatIRVisitor\n";
    ev->visit((const LocToLoc *)this);
}
template<>
void ExprNode<EqSatExtensions::GLoad>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "GLoad can only be mutated by EqSatIRVisitor\n";
    ev->visit((const GLoad *)this);
}
template<>
void ExprNode<EqSatExtensions::Computed>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "Computed can only be mutated by EqSatIRVisitor\n";
    ev->visit((const Computed *)this);
}

template<>
void ExprNode<EqSatExtensions::GVariable>::accept(IRVisitor *v) const {
    using namespace EqSatExtensions;
    EqSatIRVisitor *ev = (EqSatIRVisitor *)v;
    internal_assert(!v->is_base_ir_visitor()) << "GVariable can only be mutated by EqSatIRVisitor\n";
    ev->visit((const GVariable *)this);
}

template<>
Expr ExprNode<EqSatExtensions::LocToLoc>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "LocToLoc can only be mutated by EqSatIRMutator\n";
    return ev->visit((const LocToLoc *)this);
}

template<>
Expr ExprNode<EqSatExtensions::GLoad>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "GLoad can only be mutated by EqSatIRMutator\n";
    return ev->visit((const GLoad *)this);
}

template<>
Expr ExprNode<EqSatExtensions::GVariable>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "GVariable can only be mutated by EqSatIRMutator\n";
    return ev->visit((const GVariable *)this);
}

template<>
Expr ExprNode<EqSatExtensions::Computed>::mutate_expr(IRMutator *v) const {
    using namespace EqSatExtensions;
    EqSatIRMutator *ev = (EqSatIRMutator *)v;
    internal_assert(!v->is_base_ir_mutator()) << "Computed can only be mutated by EqSatIRMutator\n";
    return ev->visit((const Computed *)this);
}

}  // namespace Internal
}  // namespace Halide
