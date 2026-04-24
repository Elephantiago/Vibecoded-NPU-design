// -----------------------------------------------------------------------------
// Hyperion Matrix Vortex - Exascale AI Supercomputer Node v21.1
// v21.1: stride-correct TMA, closed-loop KV pager, and FP-scale quantization
// Adds descriptor-driven Tensor Memory Accelerator (TMA), lightweight KV pager,
// per-tensor/per-channel quantization hooks, LUT-style RoPE scaffold, and
// stronger kernel-test scaffolding.
// -----------------------------------------------------------------------------
// This file is a self-contained SystemVerilog baseline that merges the v15
// platform features with the v17 compute/VPU direction, then fixes the issues
// found in the static review and architecture-feedback passes:
//   * real ready/valid hold behavior on east/south egresses;
//   * input FIFO pops gated by core acceptance;
//   * north/west AllReduce operand delay alignment;
//   * safer FIFO pointer wrap for non-power-of-two depths;
//   * explicit parameter assertions for currently hard 16/32-bit datapaths;
//   * PE control/data alignment for the two-cycle Omni-MAC;
//   * OS-mode accumulator forwarding;
//   * driven FlashAttention state, readout, l/out scaling, and V input path;
//   * non-destructive RoPE placeholder instead of bit corruption;
//   * clearer comments around approximations and non-production FP math;
//   * native MX block INT32 accumulation/finalization path;
//   * dual-issue memory/compute scoreboard;
//   * standalone VC flit router shell for packetized torus fabrics;
//   * verification hooks and lightweight SV assertions for v20 bring-up;
//   * v20.1: per-VC router queues, round-robin arbitration, fixed dual-issue
//     queue accounting, MX finalize assertions, and expanded CRV scaffold;
//   * v20.2: single-assignment router credit updates, exact packet-count
//     verification hooks, explicit dual-lane sequencer debug/status, a clearer
//     native-MX INT32 bypass/finalize split, optional FP32 south AllReduce, and
//     FlashAttention RTL-vs-Python tolerance-test hooks.
//
// Important production caveat:
//   The FP32/exp/reciprocal blocks are compact deterministic reference RTL, not
//   IEEE-754-complete hardened IP. They intentionally handle finite normal/zero
//   values and basic Inf/NaN pass-through only. Replace them for silicon.
// -----------------------------------------------------------------------------

`timescale 1ns / 1ps

// -----------------------------------------------------------------------------
// Synchronous FIFO
// -----------------------------------------------------------------------------
module sync_fifo #(
    parameter int DATA_W = 64,
    parameter int DEPTH  = 64,
    parameter int ALMOST_FULL_THRESH = (DEPTH*3)/4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic push,
    input  logic [DATA_W-1:0] data_in,
    input  logic pop,
    output logic [DATA_W-1:0] data_out,
    output logic valid_out,
    output logic empty,
    output logic full,
    output logic almost_full
);
    localparam int PTR_W   = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
    localparam int COUNT_W = PTR_W + 1;
    localparam logic [COUNT_W-1:0] DEPTH_COUNT = DEPTH;
    localparam logic [COUNT_W-1:0] AFULL_COUNT = ALMOST_FULL_THRESH;
    localparam logic [PTR_W-1:0] LAST_PTR = DEPTH-1;
    localparam logic [PTR_W-1:0] PTR_ONE = 1;

    logic [DATA_W-1:0] mem [0:DEPTH-1];
    logic [PTR_W-1:0]  wr_ptr, rd_ptr;
    logic [COUNT_W-1:0] count;

    assign empty       = (count == '0);
    assign full        = (count == DEPTH_COUNT);
    assign valid_out   = !empty;
    assign almost_full = (count >= AFULL_COUNT);
    assign data_out    = mem[rd_ptr];

    function automatic logic [PTR_W-1:0] ptr_inc(input logic [PTR_W-1:0] ptr);
        begin
            ptr_inc = (ptr == LAST_PTR) ? '0 : (ptr + PTR_ONE);
        end
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            wr_ptr <= '0;
            rd_ptr <= '0;
            count  <= '0;
        end else begin
            unique case ({push && !full, pop && !empty})
                2'b10: begin
                    mem[wr_ptr] <= data_in;
                    wr_ptr <= ptr_inc(wr_ptr);
                    count  <= count + 1'b1;
                end
                2'b01: begin
                    rd_ptr <= ptr_inc(rd_ptr);
                    count  <= count - 1'b1;
                end
                2'b11: begin
                    mem[wr_ptr] <= data_in;
                    wr_ptr <= ptr_inc(wr_ptr);
                    rd_ptr <= ptr_inc(rd_ptr);
                end
                default: begin end
            endcase
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (DEPTH > 0) else $fatal("sync_fifo DEPTH must be positive");
        assert (ALMOST_FULL_THRESH <= DEPTH) else $fatal("ALMOST_FULL_THRESH must be <= DEPTH");
    end
    property p_no_overflow_push_drop;
        @(posedge clk) disable iff (!rst_n)
            !(push && full && !(pop && !empty));
    endproperty
    property p_no_underflow_pop;
        @(posedge clk) disable iff (!rst_n)
            !(pop && empty);
    endproperty
    assert property (p_no_overflow_push_drop) else $error("sync_fifo push while full would drop data");
    assert property (p_no_underflow_pop) else $error("sync_fifo pop while empty");
`endif
endmodule

// -----------------------------------------------------------------------------
// One-entry ready/valid output register.
// Holds data stable while valid=1 and downstream ready=0.
// -----------------------------------------------------------------------------
module axis_hold_reg #(
    parameter int DATA_W = 64
)(
    input  logic clk,
    input  logic rst_n,
    input  logic ce,
    input  logic [DATA_W-1:0] s_data,
    input  logic s_valid,
    output logic s_ready,
    output logic [DATA_W-1:0] m_data,
    output logic m_valid,
    input  logic m_ready
);
    assign s_ready = !m_valid || m_ready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_data  <= '0;
            m_valid <= 1'b0;
        end else if (ce && s_ready) begin
            m_data  <= s_data;
            m_valid <= s_valid;
        end else if (m_valid && m_ready) begin
            m_valid <= 1'b0;
        end
    end
`ifndef SYNTHESIS
    property p_hold_stable_when_backpressured;
        @(posedge clk) disable iff (!rst_n)
            (m_valid && !m_ready) |=> (m_valid && $stable(m_data));
    endproperty
    assert property (p_hold_stable_when_backpressured)
        else $error("axis_hold_reg changed data or dropped valid while backpressured");
`endif
endmodule

// -----------------------------------------------------------------------------
// Compact finite FP32 helper functions/modules.
// These are deterministic reference blocks, not full IEEE-754 IP.
// -----------------------------------------------------------------------------
module fp32_adder (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] sum
);
    logic sign_a, sign_b, sign_big, sign_small, sign_out;
    logic [7:0] exp_a, exp_b, exp_big, exp_small, exp_out, exp_diff;
    logic [24:0] mant_a, mant_b, mant_big, mant_small, mant_small_shifted, mant_norm;
    logic [25:0] mant_calc;
    logic [4:0]  norm_shift;

    // Single-shift normalization avoids an iterative shift/subtract loop in the
    // subtract/cancellation path. This is still compact reference FP, not a fully
    // rounded IEEE-754 implementation.
    function automatic logic [4:0] leading_zero_shift_23(input logic [24:0] mant);
        begin
            if      (mant[23]) leading_zero_shift_23 = 5'd0;
            else if (mant[22]) leading_zero_shift_23 = 5'd1;
            else if (mant[21]) leading_zero_shift_23 = 5'd2;
            else if (mant[20]) leading_zero_shift_23 = 5'd3;
            else if (mant[19]) leading_zero_shift_23 = 5'd4;
            else if (mant[18]) leading_zero_shift_23 = 5'd5;
            else if (mant[17]) leading_zero_shift_23 = 5'd6;
            else if (mant[16]) leading_zero_shift_23 = 5'd7;
            else if (mant[15]) leading_zero_shift_23 = 5'd8;
            else if (mant[14]) leading_zero_shift_23 = 5'd9;
            else if (mant[13]) leading_zero_shift_23 = 5'd10;
            else if (mant[12]) leading_zero_shift_23 = 5'd11;
            else if (mant[11]) leading_zero_shift_23 = 5'd12;
            else if (mant[10]) leading_zero_shift_23 = 5'd13;
            else if (mant[9])  leading_zero_shift_23 = 5'd14;
            else if (mant[8])  leading_zero_shift_23 = 5'd15;
            else if (mant[7])  leading_zero_shift_23 = 5'd16;
            else if (mant[6])  leading_zero_shift_23 = 5'd17;
            else if (mant[5])  leading_zero_shift_23 = 5'd18;
            else if (mant[4])  leading_zero_shift_23 = 5'd19;
            else if (mant[3])  leading_zero_shift_23 = 5'd20;
            else if (mant[2])  leading_zero_shift_23 = 5'd21;
            else if (mant[1])  leading_zero_shift_23 = 5'd22;
            else if (mant[0])  leading_zero_shift_23 = 5'd23;
            else               leading_zero_shift_23 = 5'd24;
        end
    endfunction

    always_comb begin
        sign_a = a[31];
        sign_b = b[31];
        exp_a  = a[30:23];
        exp_b  = b[30:23];
        mant_a = (exp_a == 8'd0) ? 25'd0 : {2'b01, a[22:0]};
        mant_b = (exp_b == 8'd0) ? 25'd0 : {2'b01, b[22:0]};

        sign_big = sign_a;
        sign_small = sign_b;
        exp_big = exp_a;
        exp_small = exp_b;
        mant_big = mant_a;
        mant_small = mant_b;
        if ((exp_b > exp_a) || ((exp_b == exp_a) && (mant_b > mant_a))) begin
            sign_big = sign_b;
            sign_small = sign_a;
            exp_big = exp_b;
            exp_small = exp_a;
            mant_big = mant_b;
            mant_small = mant_a;
        end

        exp_diff = exp_big - exp_small;
        mant_small_shifted = mant_small >> ((exp_diff > 8'd24) ? 5'd24 : exp_diff[4:0]);
        exp_out = exp_big;
        sign_out = sign_big;
        mant_calc = 26'd0;
        mant_norm = 25'd0;
        norm_shift = 5'd0;
        sum = 32'd0;

        if (a[30:23] == 8'hFF) begin
            sum = a;
        end else if (b[30:23] == 8'hFF) begin
            sum = b;
        end else if (a[30:0] == 31'd0) begin
            sum = b;
        end else if (b[30:0] == 31'd0) begin
            sum = a;
        end else if (sign_big == sign_small) begin
            mant_calc = {1'b0, mant_big} + {1'b0, mant_small_shifted};
            if (mant_calc[24]) begin
                mant_norm = mant_calc[24:0] >> 1;
                exp_out = exp_big + 8'd1;
            end else begin
                mant_norm = mant_calc[24:0];
            end
            sum = (exp_out == 8'hFF) ? {sign_out, 8'hFE, 23'h7FFFFF} : {sign_out, exp_out, mant_norm[22:0]};
        end else begin
            mant_calc = {1'b0, mant_big} - {1'b0, mant_small_shifted};
            mant_norm = mant_calc[24:0];
            norm_shift = leading_zero_shift_23(mant_norm);
            if ((mant_norm != 25'd0) && (norm_shift != 5'd0)) begin
                if (exp_out > norm_shift) begin
                    mant_norm = mant_norm << norm_shift;
                    exp_out = exp_out - norm_shift;
                end else begin
                    mant_norm = 25'd0;
                    exp_out = 8'd0;
                end
            end
            sum = (mant_norm == 25'd0) ? 32'd0 : {sign_out, exp_out, mant_norm[22:0]};
        end
    end
endmodule

module fp32_sub (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] diff
);
    fp32_adder u_sub (.a(a), .b({~b[31], b[30:0]}), .sum(diff));
endmodule

module fp32_mul (
    input  logic [31:0] a,
    input  logic [31:0] b,
    output logic [31:0] prod
);
    logic sign_out;
    logic signed [10:0] exp_unbiased, exp_norm;
    logic [47:0] mant_product;
    logic [22:0] frac_out;

    always_comb begin
        sign_out = a[31] ^ b[31];
        exp_unbiased = $signed({3'b000, a[30:23]}) + $signed({3'b000, b[30:23]}) - 11'sd127;
        mant_product = {1'b1, a[22:0]} * {1'b1, b[22:0]};
        exp_norm = exp_unbiased;
        frac_out = 23'd0;
        prod = 32'd0;

        if ((a[30:23] == 8'hFF) || (b[30:23] == 8'hFF)) begin
            prod = {sign_out, 8'hFF, 23'd0};
        end else if ((a[30:0] == 31'd0) || (b[30:0] == 31'd0) || (a[30:23] == 8'd0) || (b[30:23] == 8'd0)) begin
            prod = 32'd0;
        end else begin
            if (mant_product[47]) begin
                exp_norm = exp_unbiased + 11'sd1;
                frac_out = mant_product[46:24];
            end else begin
                frac_out = mant_product[45:23];
            end

            if (exp_norm <= 11'sd0) prod = 32'd0;
            else if (exp_norm >= 11'sd255) prod = {sign_out, 8'hFE, 23'h7FFFFF};
            else prod = {sign_out, exp_norm[7:0], frac_out};
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Approximate FP32 reciprocal using a single Newton-Raphson-style refinement.
// This replaces per-column FP division in the FlashAttention path. It contains
// no synthesizer "/" operator. Production silicon should use a table-based seed
// plus hardened FP multiply/subtract stages.
// -----------------------------------------------------------------------------
module fp32_recip_nr_approx (
    input  logic [31:0] x,
    output logic [31:0] recip
);
    localparam logic [31:0] FP32_TWO = 32'h4000_0000;

    logic [31:0] y0;
    logic [31:0] xy0, two_minus_xy0, y1;
    logic [31:0] xy1, two_minus_xy1;

    // Lightweight finite-normal seed. It is still not IEEE bit-exact and should
    // be replaced with a hardened LUT seed in a production FP32/scientific part.
    // v19.1 adds a second Newton-Raphson refinement over v19 to reduce the
    // normalization error while keeping the FlashAttention backend divider-free.
    always_comb begin
        if (x[30:0] == 31'd0) begin
            y0 = {x[31], 8'hFE, 23'h7FFFFF};
        end else if (x[30:23] == 8'hFF) begin
            y0 = {x[31], 31'd0};
        end else begin
            y0 = {x[31], (8'd254 - x[30:23]), ~x[22:0]};
        end
    end

    fp32_mul u_seed_mul0 (.a(x),  .b(y0),            .prod(xy0));
    fp32_sub u_nr_sub0   (.a(FP32_TWO), .b(xy0),     .diff(two_minus_xy0));
    fp32_mul u_nr_mul0   (.a(y0), .b(two_minus_xy0), .prod(y1));

    fp32_mul u_seed_mul1 (.a(x),  .b(y1),            .prod(xy1));
    fp32_sub u_nr_sub1   (.a(FP32_TWO), .b(xy1),     .diff(two_minus_xy1));
    fp32_mul u_nr_mul1   (.a(y1), .b(two_minus_xy1), .prod(recip));
endmodule

// -----------------------------------------------------------------------------
// Shared FlashAttention normalizer.
// Columns submit numerator/denominator pairs. One shared reciprocal datapath and
// one reciprocal pipeline and one FP multiplier service pending columns round-robin, replacing N per-column
// dividers with a single amortized normalization backend.
// -----------------------------------------------------------------------------
module flash_shared_normalizer #(
    parameter int COLS = 4,
    parameter int PS_W = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic ce,
    input  logic clear,
    input  logic [PS_W-1:0] num_in [0:COLS-1],
    input  logic [PS_W-1:0] den_in [0:COLS-1],
    input  logic valid_in [0:COLS-1],
    output logic [PS_W-1:0] norm_out [0:COLS-1],
    output logic valid_out [0:COLS-1]
);
    localparam int IDX_W = (COLS <= 1) ? 1 : $clog2(COLS);
    localparam logic [31:0] FP32_ONE = 32'h3F80_0000;

    logic [PS_W-1:0] pend_num [0:COLS-1];
    logic [PS_W-1:0] pend_den [0:COLS-1];
    logic pending [0:COLS-1];
    logic [IDX_W-1:0] rr_ptr;

    logic [IDX_W-1:0] sel_idx_comb;
    logic sel_valid_comb;
    logic [PS_W-1:0] sel_num_comb, sel_den_comb;

    // Stage 4A/4B/4C of FlashAttention normalization backend:
    //   4A: select a pending column request,
    //   4B: compute reciprocal approximation of denominator,
    //   4C: multiply numerator by reciprocal and write back the selected column.
    logic s4a_valid, s4b_valid, s4c_valid;
    logic [IDX_W-1:0] s4a_idx, s4b_idx, s4c_idx;
    logic [PS_W-1:0] s4a_num, s4a_den, s4a_recip_comb, s4b_num, s4b_recip, norm_comb;

    function automatic logic [IDX_W-1:0] idx_inc(input logic [IDX_W-1:0] idx);
        idx_inc = (idx == COLS-1) ? '0 : idx + 1'b1;
    endfunction

    always_comb begin
        sel_idx_comb = rr_ptr;
        sel_valid_comb = 1'b0;
        for (int k = 0; k < COLS; k++) begin
            int unsigned probe;
            probe = (rr_ptr + k) % COLS;
            if (!sel_valid_comb && pending[probe]) begin
                sel_idx_comb = probe[IDX_W-1:0];
                sel_valid_comb = 1'b1;
            end
        end
        sel_num_comb = pend_num[sel_idx_comb];
        sel_den_comb = (pend_den[sel_idx_comb][30:0] == 31'd0) ? FP32_ONE : pend_den[sel_idx_comb];
    end

    fp32_recip_nr_approx u_recip (.x(s4a_den), .recip(s4a_recip_comb));
    fp32_mul u_norm_mul (.a(s4b_num), .b(s4b_recip), .prod(norm_comb));

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rr_ptr <= '0;
            s4a_valid <= 1'b0;
            s4b_valid <= 1'b0;
            s4c_valid <= 1'b0;
            s4a_idx <= '0;
            s4b_idx <= '0;
            s4c_idx <= '0;
            s4a_num <= '0;
            s4a_den <= FP32_ONE;
            s4b_num <= '0;
            s4b_recip <= FP32_ONE;
            for (int c = 0; c < COLS; c++) begin
                pending[c]   <= 1'b0;
                pend_num[c]  <= '0;
                pend_den[c]  <= FP32_ONE;
                norm_out[c]  <= '0;
                valid_out[c] <= 1'b0;
            end
        end else if (ce) begin
            for (int c = 0; c < COLS; c++) begin
                valid_out[c] <= 1'b0;
                if (valid_in[c] && !pending[c]) begin
                    pend_num[c] <= num_in[c];
                    pend_den[c] <= (den_in[c][30:0] == 31'd0) ? FP32_ONE : den_in[c];
                    pending[c]  <= 1'b1;
                end
            end

            if (clear) begin
                rr_ptr <= '0;
                s4a_valid <= 1'b0;
                s4b_valid <= 1'b0;
                s4c_valid <= 1'b0;
                for (int c = 0; c < COLS; c++) begin
                    pending[c]   <= 1'b0;
                    valid_out[c] <= 1'b0;
                    norm_out[c]  <= '0;
                end
            end else begin
                // Stage 4C writeback.
                if (s4c_valid) begin
                    norm_out[s4c_idx]  <= norm_comb;
                    valid_out[s4c_idx] <= 1'b1;
                end

                // Pipeline advance: 4B -> 4C, 4A -> 4B.
                s4c_valid <= s4b_valid;
                s4c_idx   <= s4b_idx;
                s4b_valid <= s4a_valid;
                s4b_idx   <= s4a_idx;
                s4b_num   <= s4a_num;
                s4b_recip <= s4a_recip_comb;

                // New selection for 4A. Mark selected request consumed so the
                // round-robin backend does not re-select it while it is in pipe.
                s4a_valid <= sel_valid_comb;
                s4a_idx   <= sel_idx_comb;
                s4a_num   <= sel_num_comb;
                s4a_den   <= sel_den_comb;
                if (sel_valid_comb) begin
                    pending[sel_idx_comb] <= 1'b0;
                    rr_ptr <= idx_inc(sel_idx_comb);
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (COLS > 0) else $fatal("flash_shared_normalizer COLS must be positive");
        assert (PS_W == 32) else $fatal("flash_shared_normalizer currently requires PS_W=32");
    end
`endif
endmodule

// -----------------------------------------------------------------------------
// Registered clock-enable relay.
// The root step signal is captured once, then fanned out as local row/column
// enables. Physical implementation can place the relay flops near their sinks or
// replace this wrapper with credit/skid relays without changing core ports.
// -----------------------------------------------------------------------------
module ce_relay_grid #(
    parameter int ROWS = 4,
    parameter int COLS = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic root_step,
    output logic ingress_ce,
    output logic row_ce [0:ROWS-1],
    output logic col_ce [0:COLS-1]
);
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            ingress_ce <= 1'b0;
            for (int r = 0; r < ROWS; r++) row_ce[r] <= 1'b0;
            for (int c = 0; c < COLS; c++) col_ce[c] <= 1'b0;
        end else begin
            ingress_ce <= root_step;
            for (int r = 0; r < ROWS; r++) row_ce[r] <= root_step;
            for (int c = 0; c < COLS; c++) col_ce[c] <= root_step;
        end
    end
endmodule

// -----------------------------------------------------------------------------
// Ping-pong vector TCSM feed wrapper.
// One bank can be filled while the other feeds the systolic array. The wide raw
// top-level buses are now load-data ports into local storage rather than direct
// every-cycle array inputs.
// -----------------------------------------------------------------------------
module ping_pong_vector_tcsm #(
    parameter int DATA_W = 64,
    parameter int DEPTH  = 256,
    parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic load_en,
    input  logic load_bank,
    input  logic [ADDR_W-1:0] load_addr,
    input  logic [DATA_W-1:0] load_data,
    input  logic [ADDR_W-1:0] read_addr,
    input  logic swap_banks,
    output logic [DATA_W-1:0] read_data,
    output logic active_bank
);
    logic [DATA_W-1:0] bank0 [0:DEPTH-1];
    logic [DATA_W-1:0] bank1 [0:DEPTH-1];

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            active_bank <= 1'b0;
            read_data   <= '0;
        end else begin
            if (load_en) begin
                if (load_bank) bank1[load_addr] <= load_data;
                else           bank0[load_addr] <= load_data;
            end
            if (swap_banks) active_bank <= ~active_bank;
            read_data <= active_bank ? bank1[read_addr] : bank0[read_addr];
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (DEPTH > 0) else $fatal("ping_pong_vector_tcsm DEPTH must be positive");
    end
`endif
endmodule


// -----------------------------------------------------------------------------
// Tensor Descriptor Loader / TMA-lite
// -----------------------------------------------------------------------------
// This descriptor-driven streaming TMA front end keeps the HBM/AXI-MM side
// abstract for v21: an external memory fabric supplies one vector beat per
// tma_stream_valid. The TMA writes those beats into the selected ping-pong TCSM
// bank and autonomously increments destination addresses across a 2D tile.
// dst_kind: 0=weights, 1=V vectors, 2=sparse metadata, 3=reserved.
// -----------------------------------------------------------------------------
module tma_tensor_loader #(
    parameter int DATA_W = 128,
    parameter int ADDR_W = 8,
    parameter int DIM_W  = 16
)(
    input  logic clk,
    input  logic rst_n,
    input  logic desc_valid,
    output logic desc_ready,
    input  logic [63:0] desc_base_addr,
    input  logic [DIM_W-1:0] desc_dim_m,
    input  logic [DIM_W-1:0] desc_dim_n,
    input  logic [DIM_W-1:0] desc_stride_m,
    input  logic [DIM_W-1:0] desc_stride_n,
    input  logic [DIM_W-1:0] desc_tile_m,
    input  logic [DIM_W-1:0] desc_tile_n,
    input  logic [1:0] desc_dst_kind,
    input  logic desc_dst_bank,
    input  logic hold,
    input  logic [DATA_W-1:0] stream_data,
    input  logic stream_valid,
    output logic stream_ready,
    output logic load_valid,
    output logic [1:0] load_dst_kind,
    output logic load_bank,
    output logic [ADDR_W-1:0] load_addr,
    output logic [63:0] load_addr_full,
    output logic [DATA_W-1:0] load_data,
    output logic busy,
    output logic done,
    output logic desc_error
);
    logic [DIM_W-1:0] row_q, col_q, tile_m_q, tile_n_q;
    logic [63:0] row_base_q, addr_q;
    logic [1:0] dst_kind_q;
    logic dst_bank_q;
    logic active_q;
    logic [63:0] base_addr_q;
    logic [DIM_W-1:0] stride_m_q, stride_n_q, dim_m_q, dim_n_q;
    logic desc_bad;

    assign desc_ready   = !active_q && !hold;
    assign stream_ready = active_q && !hold;
    assign busy         = active_q;
    assign load_addr_full = addr_q;

    always_comb begin
        desc_bad = (desc_dim_m == '0) || (desc_dim_n == '0) ||
                   ((desc_tile_m != '0) && (desc_tile_m > desc_dim_m)) ||
                   ((desc_tile_n != '0) && (desc_tile_n > desc_dim_n));
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            row_q <= '0; col_q <= '0; tile_m_q <= '0; tile_n_q <= '0;
            row_base_q <= 64'd0; addr_q <= 64'd0;
            dst_kind_q <= '0; dst_bank_q <= 1'b0; active_q <= 1'b0;
            load_valid <= 1'b0; done <= 1'b0; desc_error <= 1'b0;
            base_addr_q <= '0; stride_m_q <= '0; stride_n_q <= '0; dim_m_q <= '0; dim_n_q <= '0;
            load_dst_kind <= '0; load_bank <= 1'b0; load_addr <= '0; load_data <= '0;
        end else begin
            load_valid <= 1'b0;
            done       <= 1'b0;
            desc_error <= 1'b0;
            if (desc_valid && desc_ready) begin
                if (desc_bad) begin
                    desc_error <= 1'b1;
                    active_q   <= 1'b0;
                end else begin
                    active_q   <= 1'b1;
                    row_q      <= '0;
                    col_q      <= '0;
                    row_base_q <= desc_base_addr;
                    addr_q     <= desc_base_addr;
                    tile_m_q   <= (desc_tile_m == '0) ? desc_dim_m : desc_tile_m;
                    tile_n_q   <= (desc_tile_n == '0) ? desc_dim_n : desc_tile_n;
                    dst_kind_q <= desc_dst_kind;
                    dst_bank_q <= desc_dst_bank;
                    base_addr_q <= desc_base_addr;
                    // v21.1: descriptor strides are now real geometry inputs.
                    // stride_m moves to the next row base. stride_n moves to the next column.
                    stride_m_q  <= (desc_stride_m == '0) ? desc_dim_n : desc_stride_m;
                    stride_n_q  <= (desc_stride_n == '0) ? {{(DIM_W-1){1'b0}},1'b1} : desc_stride_n;
                    dim_m_q     <= desc_dim_m;
                    dim_n_q     <= desc_dim_n;
                end
            end else if (active_q && stream_valid && stream_ready) begin
                logic end_col;
                logic end_row;
                logic [63:0] next_row_base;
                end_col = (col_q + 1'b1 >= tile_n_q);
                end_row = (row_q + 1'b1 >= tile_m_q);
                next_row_base = row_base_q + {{(64-DIM_W){1'b0}}, stride_m_q};

                load_valid    <= 1'b1;
                load_dst_kind <= dst_kind_q;
                load_bank     <= dst_bank_q;
                load_addr     <= addr_q[ADDR_W-1:0];
                load_data     <= stream_data;

                if (end_col) begin
                    col_q <= '0;
                    if (end_row) begin
                        row_q    <= '0;
                        active_q <= 1'b0;
                        done     <= 1'b1;
                    end else begin
                        row_q      <= row_q + 1'b1;
                        row_base_q <= next_row_base;
                        addr_q     <= next_row_base;
                    end
                end else begin
                    col_q  <= col_q + 1'b1;
                    addr_q <= addr_q + {{(64-DIM_W){1'b0}}, stride_n_q};
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (DATA_W > 0) else $fatal("TMA DATA_W must be positive");
        assert (ADDR_W > 0) else $fatal("TMA ADDR_W must be positive");
    end
    property p_tma_hold_blocks_load;
        @(posedge clk) disable iff (!rst_n) hold |-> !load_valid;
    endproperty
    assert property (p_tma_hold_blocks_load) else $error("TMA emitted load while held by pager/fault logic");
`endif
endmodule

// -----------------------------------------------------------------------------
// Lightweight KV Page Table Walker// -----------------------------------------------------------------------------
// Lightweight KV Page Table Walker
// -----------------------------------------------------------------------------
// Maps virtual KV token/page ids to physical page base addresses. This is a v21
// PagedAttention scaffold: deterministic translation and miss reporting, with
// replacement policy and multi-tenant permissions left for v22+.
// -----------------------------------------------------------------------------
module kv_page_table #(
    parameter int VPN_W      = 12,
    parameter int PPN_W      = 24,
    parameter int PAGE_COUNT = 256,
    parameter int PAGE_AW    = (PAGE_COUNT <= 1) ? 1 : $clog2(PAGE_COUNT)
)(
    input  logic clk,
    input  logic rst_n,
    input  logic lookup_valid,
    output logic lookup_ready,
    input  logic [VPN_W-1:0] lookup_vpn,
    output logic lookup_resp_valid,
    output logic lookup_miss,
    output logic [PPN_W-1:0] lookup_ppn,
    output logic pager_stall,
    output logic fault_valid,
    output logic [VPN_W-1:0] fault_vpn,
    input  logic fault_clear,
    input  logic ptw_write_valid,
    input  logic [PAGE_AW-1:0] ptw_write_index,
    input  logic [VPN_W-1:0] ptw_write_vpn,
    input  logic [PPN_W-1:0] ptw_write_ppn,
    input  logic ptw_write_valid_bit
);
    logic [VPN_W-1:0] vpn_mem [0:PAGE_COUNT-1];
    logic [PPN_W-1:0] ppn_mem [0:PAGE_COUNT-1];
    logic valid_mem [0:PAGE_COUNT-1];
    logic [PAGE_AW-1:0] idx;
    logic hit_comb;

    assign idx = lookup_vpn[PAGE_AW-1:0];
    assign hit_comb = valid_mem[idx] && (vpn_mem[idx] == lookup_vpn);
    assign pager_stall = fault_valid;
    assign lookup_ready = !pager_stall || fault_clear;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            lookup_resp_valid <= 1'b0;
            lookup_miss <= 1'b1;
            lookup_ppn <= '0;
            fault_valid <= 1'b0;
            fault_vpn <= '0;
            for (int i = 0; i < PAGE_COUNT; i++) begin
                valid_mem[i] <= 1'b0;
                vpn_mem[i] <= '0;
                ppn_mem[i] <= '0;
            end
        end else begin
            lookup_resp_valid <= 1'b0;
            if (ptw_write_valid) begin
                vpn_mem[ptw_write_index]   <= ptw_write_vpn;
                ppn_mem[ptw_write_index]   <= ptw_write_ppn;
                valid_mem[ptw_write_index] <= ptw_write_valid_bit;
            end
            if (fault_clear) begin
                fault_valid <= 1'b0;
            end
            if (lookup_valid && lookup_ready) begin
                lookup_resp_valid <= 1'b1;
                lookup_ppn  <= ppn_mem[idx];
                lookup_miss <= !hit_comb;
                if (!hit_comb) begin
                    // v21.1 closes the previously open-loop pager: a miss now
                    // becomes an architected fault/stall. Host or a future HW
                    // walker writes the missing entry, asserts fault_clear, and
                    // software/microcode retries the blocked descriptor.
                    fault_valid <= 1'b1;
                    fault_vpn   <= lookup_vpn;
                end
            end
        end
    end

`ifndef SYNTHESIS
    property p_fault_causes_stall;
        @(posedge clk) disable iff (!rst_n) fault_valid |-> pager_stall;
    endproperty
    assert property (p_fault_causes_stall) else $error("KV fault did not stall pager clients");
`endif
endmodule

// -----------------------------------------------------------------------------
// Quantization helpers// -----------------------------------------------------------------------------
// Quantization helpers
// -----------------------------------------------------------------------------
// v21.1 supports two post-accumulate quantization modes for integer/quantized MAC modes.
//   cfg_quant_scale_mode=0: disabled, return INT32 accumulator.
//   cfg_quant_scale_mode=1: fixed Q8.8 scale + INT32 bias.
//   cfg_quant_scale_mode=2: INT32->FP32, FP32 scale multiply, INT32 bias converted to FP32.
// This keeps the low-cost fixed-point path while adding the FP scale hook used by
// mainstream ML frameworks.
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Unified Fracturable MAC
// Modes:
//   0 INT16, 1 2xINT8, 2 4xINT4, 3 FP16,
//   4 TF32-slice approximation: consumes the 16-bit payload with the BF16-like
//     packer. This is a reduced-precision TF32 proxy, not true 32-bit TF32.
//   5 MXFP8 approximation, 6 BF16, 7 W4A8,
//   8 2:4 structured sparse: a_in={w1[7:0],w0[7:0]}, b_in contains four
//     signed INT4 activation lanes, sparse_meta={idx1[1:0],idx0[1:0]}.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Unified Fracturable MAC
// Modes:
//   0 INT16, 1 2xINT8, 2 4xINT4, 3 FP16,
//   4 TF32-slice approximation: consumes the 16-bit payload with the BF16-like
//     packer. This is a reduced-precision TF32 proxy, not true 32-bit TF32.
//   5 MXFP8. With cfg_mx_native_accum=0 it uses the reference FP path. With
//     cfg_mx_native_accum=1 it accumulates signed mantissa products natively in
//     INT32 and only converts to FP32 when cfg_mx_finalize=1.
//   6 BF16, 7 W4A8,
//   8 2:4 structured sparse: a_in={w1[7:0],w0[7:0]}, b_in contains four
//     signed INT4 activation lanes, sparse_meta={idx1[1:0],idx0[1:0]}.
//   9 MXFP4 native block accumulation: four signed INT4 lane products accumulate
//     in INT32 under the shared exponent and finalize only on cfg_mx_finalize.
// -----------------------------------------------------------------------------
module unified_fracturable_mac (
    input  logic clk,
    input  logic rst_n,
    input  logic ce,
    input  logic [3:0] cfg_mode,
    input  logic cfg_mx_native_accum,
    input  logic cfg_mx_finalize,
    input  logic [7:0] shared_exp,
    input  logic cfg_quant_en,
    input  logic [1:0] cfg_quant_scale_mode,
    input  logic [15:0] quant_scale_q8_8,
    input  logic [31:0] quant_scale_fp32,
    input  logic signed [31:0] quant_bias_i32,
    input  logic signed [15:0] act_zero_point,
    input  logic signed [15:0] wt_zero_point,
    input  logic [15:0] a_in,
    input  logic [15:0] b_in,
    input  logic [3:0] sparse_meta,
    input  logic [31:0] c_accum,
    output logic [31:0] mac_out
);
    logic [3:0] cfg_mode_q;
    logic cfg_mx_native_q, cfg_mx_finalize_q;
    logic [7:0] shared_exp_q;
    logic [31:0] c_accum_q;
    logic signed [31:0] int16_prod_q, sum_2x8_q, sum_4x4_q, w4a8_prod_q, sparse_prod_q;
    logic signed [31:0] mx8_mant_prod_q, mx4_mant_prod_q, mx_native_sum_q;
    logic signed [31:0] mx8_mant_prod_comb, mx4_mant_prod_comb;
    logic [31:0] float_product_q;
    logic [1:0] cfg_quant_scale_mode_q;
    logic cfg_quant_en_q;
    logic [15:0] quant_scale_q8_8_q;
    logic [31:0] quant_scale_fp32_q;
    logic signed [31:0] quant_bias_i32_q;
    logic signed [31:0] int_accum_selected;
    logic [31:0] int_accum_fp32;
    logic [31:0] quant_bias_fp32;
    logic [31:0] quant_fp_scaled;
    logic [31:0] quant_fp_out;

    function automatic logic signed [31:0] sxmul4(input logic [3:0] x, input logic [3:0] y);
        sxmul4 = $signed(x) * $signed(y);
    endfunction

    function automatic logic signed [31:0] sxmul8(input logic [7:0] x, input logic [7:0] y);
        sxmul8 = $signed(x) * $signed(y);
    endfunction

    function automatic logic signed [31:0] qscale_i32(input logic signed [31:0] val);
        logic signed [47:0] prod;
        begin
            prod = $signed(val) * $signed({1'b0, quant_scale_q8_8_q});
            qscale_i32 = (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd1)) ? $signed(prod >>> 8) + quant_bias_i32_q : val;
        end
    endfunction

    function automatic logic signed [15:0] zp_sub16(input logic [15:0] x, input logic signed [15:0] zp);
        zp_sub16 = $signed(x) - zp;
    endfunction

    function automatic logic signed [7:0] zp_sub8(input logic [7:0] x, input logic signed [15:0] zp);
        zp_sub8 = $signed(x) - $signed(zp[7:0]);
    endfunction

    function automatic logic signed [3:0] zp_sub4(input logic [3:0] x, input logic signed [15:0] zp);
        zp_sub4 = $signed(x) - $signed(zp[3:0]);
    endfunction

    function automatic logic signed [31:0] sxmul4x8(input logic [3:0] x, input logic [7:0] y);
        sxmul4x8 = $signed(x) * $signed(y);
    endfunction

    function automatic logic [31:0] int32_scaled_to_fp32(input logic signed [31:0] val, input logic [7:0] block_exp);
        logic sign;
        logic [31:0] mag;
        logic [5:0] msb;
        logic [55:0] shifted;
        logic signed [10:0] exp32;
        begin
            sign = val[31];
            mag  = sign ? (~val + 32'd1) : val;
            msb  = 6'd0;
            for (int k = 0; k < 32; k++) begin
                if (mag[k]) msb = k[5:0];
            end
            if (mag == 32'd0) begin
                int32_scaled_to_fp32 = 32'd0;
            end else begin
                shifted = {24'd0, mag} << (6'd31 - msb);
                // The native MX accumulator is an integer mantissa sum. MXFP8 and MXFP4
                // use this same finalizer after mode-specific mantissa products. The block
                // exponent supplies the shared scale. The msb term renormalizes the
                // integer accumulator to a FP32 mantissa. Exact OCP MX metadata has
                // more format-specific bias rules; this is the hardware-friendly
                // block-accumulation datapath with an explicit final conversion point.
                exp32 = $signed({3'd0, block_exp}) + $signed({5'd0, msb});
                if (exp32 <= 0) int32_scaled_to_fp32 = 32'd0;
                else if (exp32 >= 255) int32_scaled_to_fp32 = {sign, 8'hFE, 23'h7FFFFF};
                else int32_scaled_to_fp32 = {sign, exp32[7:0], shifted[30:8]};
            end
        end
    endfunction

    function automatic logic [31:0] int32_to_fp32(input logic signed [31:0] val);
        logic sign;
        logic [31:0] mag;
        logic [5:0] msb;
        logic [55:0] shifted;
        logic [7:0] exp32;
        begin
            sign = val[31];
            mag  = sign ? (~val + 32'd1) : val;
            msb  = 6'd0;
            for (int k = 0; k < 32; k++) begin
                if (mag[k]) msb = k[5:0];
            end
            if (mag == 32'd0) begin
                int32_to_fp32 = 32'd0;
            end else begin
                shifted = {24'd0, mag} << (6'd31 - msb);
                exp32 = 8'd127 + msb[7:0];
                int32_to_fp32 = {sign, exp32, shifted[30:8]};
            end
        end
    endfunction

    function automatic logic [31:0] pack_fp16_product(input logic [15:0] aa, input logic [15:0] bb);
        logic sign; logic [4:0] ea, eb; logic [21:0] mp; logic signed [10:0] exp32; logic [9:0] frac10;
        begin
            sign = aa[15] ^ bb[15]; ea = aa[14:10]; eb = bb[14:10];
            mp = {1'b1, aa[9:0]} * {1'b1, bb[9:0]};
            exp32 = $signed({6'd0, ea}) + $signed({6'd0, eb}) - 11'sd15 + 11'sd127;
            frac10 = 10'd0;
            if ((ea == 5'd0) || (eb == 5'd0)) pack_fp16_product = 32'd0;
            else begin
                if (mp[21]) begin exp32 = exp32 + 11'sd1; frac10 = mp[20:11]; end
                else frac10 = mp[19:10];
                pack_fp16_product = (exp32 <= 0) ? 32'd0 : {sign, exp32[7:0], frac10, 13'd0};
            end
        end
    endfunction

    function automatic logic [31:0] pack_bf16like_product(input logic [15:0] aa, input logic [15:0] bb, input logic [7:0] exp_override, input logic use_exp_override);
        logic sign; logic [7:0] ea, eb; logic signed [10:0] exp32; logic [15:0] mp; logic [6:0] frac7;
        begin
            sign = aa[15] ^ bb[15]; ea = aa[14:7]; eb = bb[14:7];
            mp = {1'b1, aa[6:0]} * {1'b1, bb[6:0]};
            exp32 = use_exp_override ? $signed({3'd0, exp_override}) : ($signed({3'd0, ea}) + $signed({3'd0, eb}) - 11'sd127);
            frac7 = 7'd0;
            if (((ea == 8'd0) || (eb == 8'd0)) && !use_exp_override) pack_bf16like_product = 32'd0;
            else begin
                if (mp[15]) begin exp32 = exp32 + 11'sd1; frac7 = mp[14:8]; end
                else frac7 = mp[13:7];
                pack_bf16like_product = (exp32 <= 0) ? 32'd0 : {sign, exp32[7:0], frac7, 16'd0};
            end
        end
    endfunction

    always_comb begin
        mx8_mant_prod_comb = sxmul8(a_in[7:0], b_in[7:0]) + sxmul8(a_in[15:8], b_in[15:8]);
        mx4_mant_prod_comb = sxmul4(a_in[3:0], b_in[3:0]) + sxmul4(a_in[7:4], b_in[7:4]) +
                              sxmul4(a_in[11:8], b_in[11:8]) + sxmul4(a_in[15:12], b_in[15:12]);
    end

    logic [1:0] meta_idx_0, meta_idx_1;
    logic signed [7:0] sp_w0, sp_w1, sp_a0, sp_a1;
    logic signed [7:0] a_lane [0:3];
    assign meta_idx_0 = sparse_meta[1:0];
    assign meta_idx_1 = sparse_meta[3:2];
    assign sp_w0 = a_in[7:0];
    assign sp_w1 = a_in[15:8];
    assign a_lane[0] = {{4{b_in[3]}},  b_in[3:0]};
    assign a_lane[1] = {{4{b_in[7]}},  b_in[7:4]};
    assign a_lane[2] = {{4{b_in[11]}}, b_in[11:8]};
    assign a_lane[3] = {{4{b_in[15]}}, b_in[15:12]};
    assign sp_a0 = a_lane[meta_idx_0];
    assign sp_a1 = a_lane[meta_idx_1];

    logic [31:0] float_adder_out;
    fp32_adder u_float_acc (.a(float_product_q), .b(c_accum_q), .sum(float_adder_out));
    fp32_mul   u_quant_fp_mul (.a(int_accum_fp32), .b(quant_scale_fp32_q), .prod(quant_fp_scaled));
    fp32_adder u_quant_fp_add (.a(quant_fp_scaled), .b(quant_bias_fp32), .sum(quant_fp_out));

    always_comb begin
        unique case (cfg_mode_q)
            4'h0: int_accum_selected = $signed(c_accum_q) + int16_prod_q;
            4'h1: int_accum_selected = $signed(c_accum_q) + sum_2x8_q;
            4'h2: int_accum_selected = $signed(c_accum_q) + sum_4x4_q;
            4'h7: int_accum_selected = $signed(c_accum_q) + w4a8_prod_q;
            4'h8: int_accum_selected = $signed(c_accum_q) + sparse_prod_q;
            default: int_accum_selected = $signed(c_accum_q);
        endcase
        int_accum_fp32 = int32_to_fp32(int_accum_selected);
        quant_bias_fp32 = int32_to_fp32(quant_bias_i32_q);
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            cfg_mode_q <= 4'd0; cfg_mx_native_q <= 1'b0; cfg_mx_finalize_q <= 1'b0; shared_exp_q <= 8'd127;
            c_accum_q <= 32'd0; int16_prod_q <= 32'sd0; sum_2x8_q <= 32'sd0; sum_4x4_q <= 32'sd0;
            w4a8_prod_q <= 32'sd0; sparse_prod_q <= 32'sd0; mx8_mant_prod_q <= 32'sd0; mx4_mant_prod_q <= 32'sd0;
            mx_native_sum_q <= 32'sd0; float_product_q <= 32'd0; cfg_quant_scale_mode_q <= 2'd0; cfg_quant_en_q <= 1'b0; quant_scale_q8_8_q <= 16'd256; quant_scale_fp32_q <= 32'h3F80_0000; quant_bias_i32_q <= 32'sd0;
        end else if (ce) begin
            cfg_mode_q <= cfg_mode;
            cfg_mx_native_q <= cfg_mx_native_accum;
            cfg_mx_finalize_q <= cfg_mx_finalize;
            shared_exp_q <= shared_exp;
            cfg_quant_en_q <= cfg_quant_en;
            cfg_quant_scale_mode_q <= cfg_quant_scale_mode;
            quant_scale_q8_8_q <= quant_scale_q8_8;
            quant_scale_fp32_q <= quant_scale_fp32;
            quant_bias_i32_q <= quant_bias_i32;
            c_accum_q <= c_accum;
            int16_prod_q <= zp_sub16(a_in, wt_zero_point) * zp_sub16(b_in, act_zero_point);
            sum_2x8_q <= (zp_sub8(a_in[7:0], wt_zero_point) * zp_sub8(b_in[7:0], act_zero_point)) + (zp_sub8(a_in[15:8], wt_zero_point) * zp_sub8(b_in[15:8], act_zero_point));
            sum_4x4_q <= (zp_sub4(a_in[3:0], wt_zero_point) * zp_sub4(b_in[3:0], act_zero_point)) + (zp_sub4(a_in[7:4], wt_zero_point) * zp_sub4(b_in[7:4], act_zero_point)) + (zp_sub4(a_in[11:8], wt_zero_point) * zp_sub4(b_in[11:8], act_zero_point)) + (zp_sub4(a_in[15:12], wt_zero_point) * zp_sub4(b_in[15:12], act_zero_point));
            w4a8_prod_q <= (zp_sub4(a_in[3:0], wt_zero_point) * zp_sub8(b_in[7:0], act_zero_point)) + (zp_sub4(a_in[7:4], wt_zero_point) * zp_sub8(b_in[15:8], act_zero_point));
            sparse_prod_q <= ($signed(sp_w0) * $signed(sp_a0)) + ($signed(sp_w1) * $signed(sp_a1));
            mx8_mant_prod_q <= mx8_mant_prod_comb;
            mx4_mant_prod_q <= mx4_mant_prod_comb;

            unique case (cfg_mode)
                4'h3: float_product_q <= pack_fp16_product(a_in, b_in);
                4'h4: float_product_q <= pack_bf16like_product(a_in, b_in, 8'd0, 1'b0);
                4'h5: float_product_q <= cfg_mx_native_accum ? 32'd0 : pack_bf16like_product(a_in, b_in, shared_exp, 1'b1);
                4'h6: float_product_q <= pack_bf16like_product(a_in, b_in, 8'd0, 1'b0);
                default: float_product_q <= 32'd0;
            endcase
            mx_native_sum_q <= $signed(c_accum) + ((cfg_mode == 4'h9) ? mx4_mant_prod_comb : mx8_mant_prod_comb);
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            mac_out <= 32'd0;
        end else if (ce) begin
            unique case (cfg_mode_q)
                4'h0: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out : qscale_i32($signed(c_accum_q) + int16_prod_q);
                4'h1: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out : qscale_i32($signed(c_accum_q) + sum_2x8_q);
                4'h2: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out : qscale_i32($signed(c_accum_q) + sum_4x4_q);
                4'h3, 4'h4, 4'h6: mac_out <= float_adder_out;
                4'h5: begin
                    if (cfg_mx_native_q) mac_out <= cfg_mx_finalize_q ? int32_scaled_to_fp32(mx_native_sum_q, shared_exp_q) : mx_native_sum_q;
                    else mac_out <= float_adder_out;
                end
                4'h7: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out : qscale_i32($signed(c_accum_q) + w4a8_prod_q);
                4'h8: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out : qscale_i32($signed(c_accum_q) + sparse_prod_q);
                4'h9: begin
                    // MXFP4 is native-accumulation only in this RTL: the FP formatter is bypassed.
                    mac_out <= (cfg_mx_native_q && cfg_mx_finalize_q) ? int32_scaled_to_fp32(mx_native_sum_q, shared_exp_q) : mx_native_sum_q;
                end
                default: mac_out <= c_accum_q;
            endcase
        end
    end
endmodule


// -----------------------------------------------------------------------------
// FlashAttention VPU front-end.
// This block owns the online-softmax state for one column but no longer divides.
// Instead, it emits numerator/denominator pairs for flash_shared_normalizer.
// -----------------------------------------------------------------------------
module flash_attention_vpu #(
    parameter int PS_W = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic ce,
    input  logic clear_state,
    input  logic [2:0] cfg_vpu_mode,
    input  logic [15:0] seq_i,
    input  logic [15:0] seq_j,
    input  logic [PS_W-1:0] x_in,
    input  logic x_valid,
    input  logic [PS_W-1:0] v_in,
    input  logic [PS_W-1:0] daisy_chain_in,
    output logic [PS_W-1:0] daisy_chain_out,
    output logic [PS_W-1:0] norm_num_out,
    output logic [PS_W-1:0] norm_den_out,
    output logic norm_valid_out
);
    localparam logic [31:0] FP32_ZERO    = 32'h0000_0000;
    localparam logic [31:0] FP32_ONE     = 32'h3F80_0000;
    localparam logic [31:0] FP32_HALF    = 32'h3F00_0000;
    localparam logic [31:0] FP32_QUARTER = 32'h3E80_0000;
    localparam logic [31:0] FP32_EIGHTH  = 32'h3E00_0000;

    // Persistent online-softmax state.
    logic [31:0] m_old, l_old, out_old;
    logic m_valid;

    function automatic logic fp32_gt(input logic [31:0] aa, input logic [31:0] bb);
        begin
            if (aa[31] != bb[31]) fp32_gt = !aa[31];
            else if (!aa[31]) fp32_gt = (aa[30:0] > bb[30:0]);
            else fp32_gt = (aa[30:0] < bb[30:0]);
        end
    endfunction

    function automatic logic [31:0] exp_neg_approx(input logic [31:0] neg_or_zero);
        logic [7:0] mag_exp;
        logic [22:0] mag_frac;
        begin
            mag_exp  = neg_or_zero[30:23];
            mag_frac = neg_or_zero[22:0];
            // Deterministic 8-bin finite-normal approximation for exp(x), x<=0.
            // Production should replace this with a table/polynomial approximation
            // sized for the target perplexity/accuracy budget.
            if (neg_or_zero[30:0] == 31'd0) begin
                exp_neg_approx = FP32_ONE;
            end else if (!neg_or_zero[31]) begin
                exp_neg_approx = FP32_ONE;
            end else if (mag_exp >= 8'd130) begin
                exp_neg_approx = 32'h3C80_0000;     // ~1/64
            end else if (mag_exp == 8'd129) begin
                exp_neg_approx = mag_frac[22] ? 32'h3D80_0000 : 32'h3E00_0000; // 1/16 or 1/8
            end else if (mag_exp == 8'd128) begin
                exp_neg_approx = mag_frac[22] ? 32'h3E40_0000 : FP32_EIGHTH;
            end else if (mag_exp == 8'd127) begin
                exp_neg_approx = mag_frac[22] ? FP32_QUARTER : 32'h3EA0_0000;
            end else begin
                exp_neg_approx = mag_frac[22] ? FP32_HALF : 32'h3F40_0000;
            end
        end
    endfunction

    // ---------------- Stage 1: mask, max, and difference setup ----------------
    logic s1_valid, s1_masked, s1_m_valid;
    logic [2:0] s1_mode;
    logic [31:0] s1_x, s1_v, s1_m_old, s1_l_old, s1_out_old, s1_m_new;
    logic [31:0] diff_old_comb, diff_new_comb;
    logic [31:0] s1_diff_old, s1_diff_new;
    logic [31:0] daisy_s1;

    logic masked_comb, x_gt_m_comb;
    logic [31:0] m_new_comb;
    assign masked_comb = (seq_j > seq_i);
    assign x_gt_m_comb = !m_valid || fp32_gt(x_in, m_old);
    assign m_new_comb  = (masked_comb || !x_gt_m_comb) ? m_old : x_in;

    fp32_sub u_diff_old (.a(m_old), .b(m_new_comb), .diff(diff_old_comb));
    fp32_sub u_diff_new (.a(x_in),  .b(m_new_comb), .diff(diff_new_comb));

    // ---------------- Stage 2: exp approximation ----------------
    logic s2_valid, s2_masked, s2_m_valid;
    logic [2:0] s2_mode;
    logic [31:0] s2_x, s2_v, s2_m_new, s2_l_old, s2_out_old;
    logic [31:0] s2_exp_old, s2_exp_new;
    logic [31:0] daisy_s2;

    // ---------------- Stage 3: L and output numerator summation ----------------
    logic [31:0] l_scaled_comb, l_sum_comb, out_scaled_comb, v_scaled_comb, out_sum_comb;
    logic s3_valid, s3_masked, s3_m_valid;
    logic [2:0] s3_mode;
    logic [31:0] s3_x, s3_m_new, s3_l_sum, s3_out_sum;
    logic [31:0] daisy_s3;

    fp32_mul   u_l_scale   (.a(s2_l_old),   .b(s2_exp_old), .prod(l_scaled_comb));
    fp32_adder u_l_add     (.a(l_scaled_comb), .b(s2_exp_new), .sum(l_sum_comb));
    fp32_mul   u_out_scale (.a(s2_out_old), .b(s2_exp_old), .prod(out_scaled_comb));
    fp32_mul   u_v_scale   (.a(s2_v),       .b(s2_exp_new), .prod(v_scaled_comb));
    fp32_adder u_out_add   (.a(out_scaled_comb), .b(v_scaled_comb), .sum(out_sum_comb));

    // ---------------- Stage 4: emit numerator/denominator to shared reciprocal ----------------
    logic s4_valid, s4_masked, s4_m_valid;
    logic [2:0] s4_mode;
    logic [31:0] s4_x, s4_m_new, s4_l_sum, s4_out_sum;
    logic [31:0] daisy_s4;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            m_old <= FP32_ZERO;
            l_old <= FP32_ZERO;
            out_old <= FP32_ZERO;
            m_valid <= 1'b0;
            s1_valid <= 1'b0; s2_valid <= 1'b0; s3_valid <= 1'b0; s4_valid <= 1'b0;
            daisy_s1 <= FP32_ZERO; daisy_s2 <= FP32_ZERO; daisy_s3 <= FP32_ZERO; daisy_s4 <= FP32_ZERO;
            norm_num_out <= FP32_ZERO;
            norm_den_out <= FP32_ONE;
            norm_valid_out <= 1'b0;
            daisy_chain_out <= FP32_ZERO;
        end else if (ce) begin
            norm_valid_out <= 1'b0;

            if (clear_state) begin
                m_old <= FP32_ZERO;
                l_old <= FP32_ZERO;
                out_old <= FP32_ZERO;
                m_valid <= 1'b0;
                s1_valid <= 1'b0; s2_valid <= 1'b0; s3_valid <= 1'b0; s4_valid <= 1'b0;
                daisy_s1 <= FP32_ZERO; daisy_s2 <= FP32_ZERO; daisy_s3 <= FP32_ZERO; daisy_s4 <= FP32_ZERO;
                norm_num_out <= FP32_ZERO;
                norm_den_out <= FP32_ONE;
                norm_valid_out <= 1'b0;
                daisy_chain_out <= FP32_ZERO;
            end else begin
                // Stage 4 writeback/readout. Reciprocal/divide is handled by the
                // shared pipelined normalizer immediately after this VPU stage.
                if (s4_valid) begin
                    unique case (s4_mode)
                        3'd0: begin
                            norm_num_out   <= s4_x;
                            norm_den_out   <= FP32_ONE;
                            norm_valid_out <= 1'b1;
                        end
                        3'd1: begin
                            norm_num_out   <= s4_x[31] ? FP32_ZERO : s4_x;
                            norm_den_out   <= FP32_ONE;
                            norm_valid_out <= 1'b1;
                        end
                        3'd2: begin
                            if (!s4_masked) begin
                                m_old   <= s4_m_new;
                                l_old   <= s4_l_sum;
                                out_old <= s4_out_sum;
                                m_valid <= 1'b1;
                                norm_num_out   <= s4_out_sum;
                                norm_den_out   <= (s4_l_sum[30:0] == 31'd0) ? FP32_ONE : s4_l_sum;
                                norm_valid_out <= 1'b1;
                            end
                        end
                        3'd3: begin
                            norm_num_out   <= out_old;
                            norm_den_out   <= (l_old[30:0] == 31'd0) ? FP32_ONE : l_old;
                            norm_valid_out <= m_valid;
                        end
                        default: begin
                            norm_num_out   <= s4_x;
                            norm_den_out   <= FP32_ONE;
                            norm_valid_out <= 1'b1;
                        end
                    endcase
                end
                daisy_chain_out <= daisy_s4;

                // Stage 3 -> Stage 4.
                s4_valid  <= s3_valid;
                s4_masked <= s3_masked;
                s4_m_valid <= s3_m_valid;
                s4_mode   <= s3_mode;
                s4_x      <= s3_x;
                s4_m_new  <= s3_m_new;
                s4_l_sum  <= s3_l_sum;
                s4_out_sum <= s3_out_sum;
                daisy_s4  <= daisy_s3;

                // Stage 2 -> Stage 3.
                s3_valid  <= s2_valid;
                s3_masked <= s2_masked;
                s3_m_valid <= s2_m_valid;
                s3_mode   <= s2_mode;
                s3_x      <= s2_x;
                s3_m_new  <= s2_m_new;
                s3_l_sum  <= l_sum_comb;
                s3_out_sum <= out_sum_comb;
                daisy_s3  <= daisy_s2;

                // Stage 1 -> Stage 2.
                s2_valid  <= s1_valid;
                s2_masked <= s1_masked;
                s2_m_valid <= s1_m_valid;
                s2_mode   <= s1_mode;
                s2_x      <= s1_x;
                s2_v      <= s1_v;
                s2_m_new  <= s1_m_new;
                s2_l_old  <= s1_l_old;
                s2_out_old <= s1_out_old;
                s2_exp_old <= (!s1_m_valid) ? FP32_ZERO : exp_neg_approx(s1_diff_old);
                s2_exp_new <= s1_masked ? FP32_ZERO : exp_neg_approx(s1_diff_new);
                daisy_s2  <= daisy_s1;

                // Input -> Stage 1.
                s1_valid  <= x_valid;
                s1_masked <= masked_comb;
                s1_m_valid <= m_valid;
                s1_mode   <= cfg_vpu_mode;
                s1_x      <= x_in;
                s1_v      <= v_in;
                s1_m_old  <= m_old;
                s1_l_old  <= l_old;
                s1_out_old <= out_old;
                s1_m_new  <= m_new_comb;
                s1_diff_old <= diff_old_comb;
                s1_diff_new <= diff_new_comb;
                daisy_s1  <= (cfg_vpu_mode == 3'd2) ?
                             (fp32_gt(x_in, daisy_chain_in) ? x_in : daisy_chain_in) :
                             daisy_chain_in;
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (PS_W == 32) else $fatal("flash_attention_vpu currently requires PS_W=32");
    end
`endif
endmodule

// -----------------------------------------------------------------------------
// RoPE engine placeholder.
// The prior XOR placeholder corrupted payload bits. This version preserves data
// and timing. Replace rotate_data_identity() with a true CORDIC/sin-cos lane
// rotation when real RoPE is required.
// -----------------------------------------------------------------------------
module rope_engine #(
    parameter int DATA_W = 64
)(
    input  logic clk,
    input  logic rst_n,
    input  logic cfg_rope_en,
    input  logic [DATA_W-1:0] s_tdata,
    input  logic s_tvalid,
    output logic s_tready,
    output logic [DATA_W-1:0] m_tdata,
    output logic m_tvalid,
    input  logic m_tready
);
    // v21 LUT-style RoPE scaffold.  For compactness this block rotates each
    // adjacent 16-bit pair with one of four coarse angles selected by a phase
    // counter: 0, 90, 180, 270 degrees. It is non-destructive, pipelined, and
    // structurally matches real RoPE data movement; replace the coarse LUT with
    // BF16/FP16 sin/cos ROMs for production accuracy.
    localparam int LANES = DATA_W / 16;
    logic [DATA_W-1:0] rope_pipe [0:1];
    logic valid_pipe [0:1];
    logic [1:0] phase_q;

    function automatic logic [31:0] rotate_pair(input logic [15:0] x, input logic [15:0] y, input logic [1:0] phase);
        begin
            unique case (phase)
                2'd0: rotate_pair = {y, x};
                2'd1: rotate_pair = {x, (~y + 16'd1)};
                2'd2: rotate_pair = {(~y + 16'd1), (~x + 16'd1)};
                default: rotate_pair = {(~x + 16'd1), y};
            endcase
        end
    endfunction

    assign s_tready = m_tready;

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            rope_pipe[0] <= '0;
            rope_pipe[1] <= '0;
            valid_pipe[0] <= 1'b0;
            valid_pipe[1] <= 1'b0;
            phase_q <= 2'd0;
        end else if (m_tready) begin
            rope_pipe[0] <= s_tdata;
            if (cfg_rope_en) begin
                for (int l = 0; l < LANES; l += 2) begin
                    if (l+1 < LANES) begin
                        rope_pipe[0][(l*16) +: 32] <= rotate_pair(s_tdata[(l*16) +: 16], s_tdata[((l+1)*16) +: 16], phase_q);
                    end
                end
                if (s_tvalid) phase_q <= phase_q + 2'd1;
            end
            valid_pipe[0] <= s_tvalid;
            rope_pipe[1] <= rope_pipe[0];
            valid_pipe[1] <= valid_pipe[0];
        end
    end

    assign m_tdata  = cfg_rope_en ? rope_pipe[1] : s_tdata;
    assign m_tvalid = cfg_rope_en ? valid_pipe[1] : s_tvalid;

`ifndef SYNTHESIS
    initial begin
        assert ((DATA_W % 16) == 0) else $fatal("rope_engine DATA_W must be a multiple of 16");
    end
`endif
endmodule

// -----------------------------------------------------------------------------
// OoO Micro Sequencer / ISA scoreboard skeleton.
// Opcodes:
//   1: weight shift pulse
//   2: weight swap pulse
//   3: clear/compute pulse
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Dual-Issue / Multithreaded ISA Scoreboard
// -----------------------------------------------------------------------------
// One fetch stream is split into independent memory and compute issue queues.
// The memory thread can issue DMA/weight-fill operations while the compute thread
// simultaneously issues clear/compute work, enabling overlap between ping-pong
// TCSM fills and array math. This is still a compact in-core dispatcher; real
// SoCs should replace the busy inputs with completion events from the DMA engine
// and compute scheduler.
// Opcodes:
//   1: memory thread, weight/meta shift pulse + trigger_dma
//   2: memory thread, swap ping-pong/active weights
//   3: compute thread, clear partial sums + trigger_array
//   4: memory thread, software-visible DMA trigger only
//   5: compute thread, compute trigger only
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Dual-Issue / Multithreaded ISA Scoreboard v20.1
// -----------------------------------------------------------------------------
// One fetch stream is split into independent memory and compute issue queues.
// The memory lane and compute lane can issue in the same cycle when their busy
// inputs are clear.  The implementation uses explicit next-state accounting so
// simultaneous enqueue/dequeue on one thread preserves queue depth correctly.
//
// Opcode map, ir_in[31:28]:
//   1: MEM     weight/meta shift pulse + trigger_dma
//   2: MEM     swap active ping-pong/weight bank
//   3: COMPUTE clear partial sums + trigger_array
//   4: MEM     DMA trigger only
//   5: COMPUTE array trigger only
//   6: COMPUTE MX finalize pulse is expected from cfg_mx_finalize sideband
// Other opcodes are dropped intentionally in this compact bring-up sequencer.
// -----------------------------------------------------------------------------
module ooo_micro_sequencer #(
    parameter int Q_DEPTH = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [31:0] ir_in,
    input  logic ir_valid,
    output logic shift_w_en,
    output logic swap_weights,
    output logic clear_ps_base,
    input  logic dma_busy,
    input  logic array_busy,
    output logic trigger_dma,
    output logic trigger_array,
    output logic mem_issue_valid,
    output logic compute_issue_valid,
    output logic dual_issue_valid,
    output logic [7:0] mem_queue_count,
    output logic [7:0] compute_queue_count
);
    localparam int PTR_W = (Q_DEPTH <= 1) ? 1 : $clog2(Q_DEPTH);
    localparam logic [PTR_W:0] Q_DEPTH_COUNT = Q_DEPTH;

    logic [31:0] mem_q  [0:Q_DEPTH-1];
    logic [31:0] comp_q [0:Q_DEPTH-1];
    logic [PTR_W-1:0] mem_wr, mem_rd, comp_wr, comp_rd;
    logic [PTR_W:0] mem_count, comp_count;

    logic [3:0] opcode;
    logic is_mem_op, is_comp_op;
    assign opcode = ir_in[31:28];
    assign is_mem_op  = (opcode == 4'h1) || (opcode == 4'h2) || (opcode == 4'h4);
    assign is_comp_op = (opcode == 4'h3) || (opcode == 4'h5) || (opcode == 4'h6);

    function automatic logic [PTR_W-1:0] q_inc(input logic [PTR_W-1:0] p);
        q_inc = (p == Q_DEPTH-1) ? '0 : p + 1'b1;
    endfunction

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            mem_wr <= '0; mem_rd <= '0; comp_wr <= '0; comp_rd <= '0;
            mem_count <= '0; comp_count <= '0;
            shift_w_en <= 1'b0; swap_weights <= 1'b0; clear_ps_base <= 1'b0;
            trigger_dma <= 1'b0; trigger_array <= 1'b0;
            mem_issue_valid <= 1'b0; compute_issue_valid <= 1'b0; dual_issue_valid <= 1'b0;
            mem_queue_count <= '0; compute_queue_count <= '0;
        end else begin
            logic mem_enq, comp_enq, mem_deq, comp_deq;
            logic [PTR_W:0] mem_count_next, comp_count_next;
            logic [PTR_W-1:0] mem_wr_next, mem_rd_next, comp_wr_next, comp_rd_next;

            shift_w_en <= 1'b0;
            swap_weights <= 1'b0;
            clear_ps_base <= 1'b0;
            trigger_dma <= 1'b0;
            trigger_array <= 1'b0;
            mem_issue_valid <= 1'b0;
            compute_issue_valid <= 1'b0;
            dual_issue_valid <= 1'b0;

            mem_enq  = ir_valid && is_mem_op  && (mem_count  != Q_DEPTH_COUNT);
            comp_enq = ir_valid && is_comp_op && (comp_count != Q_DEPTH_COUNT);
            mem_deq  = !dma_busy   && (mem_count  != '0);
            comp_deq = !array_busy && (comp_count != '0);

            mem_count_next  = mem_count  + mem_enq  - mem_deq;
            comp_count_next = comp_count + comp_enq - comp_deq;
            mem_wr_next = mem_wr; mem_rd_next = mem_rd;
            comp_wr_next = comp_wr; comp_rd_next = comp_rd;

            if (mem_enq) begin
                mem_q[mem_wr] <= ir_in;
                mem_wr_next = q_inc(mem_wr);
            end
            if (comp_enq) begin
                comp_q[comp_wr] <= ir_in;
                comp_wr_next = q_inc(comp_wr);
            end

            // Memory-thread issue. Independent from compute-thread issue.
            mem_issue_valid <= mem_deq;
            compute_issue_valid <= comp_deq;
            dual_issue_valid <= mem_deq && comp_deq;
            if (mem_deq) begin
                unique case (mem_q[mem_rd][31:28])
                    4'h1: begin shift_w_en <= 1'b1; trigger_dma <= 1'b1; end
                    4'h2: begin swap_weights <= 1'b1; end
                    4'h4: begin trigger_dma <= 1'b1; end
                    default: begin end
                endcase
                mem_rd_next = q_inc(mem_rd);
            end

            // Compute-thread issue. Can fire in the same cycle as memory-thread.
            if (comp_deq) begin
                unique case (comp_q[comp_rd][31:28])
                    4'h3: begin clear_ps_base <= 1'b1; trigger_array <= 1'b1; end
                    4'h5: begin trigger_array <= 1'b1; end
                    4'h6: begin trigger_array <= 1'b1; end
                    default: begin end
                endcase
                comp_rd_next = q_inc(comp_rd);
            end

            mem_wr <= mem_wr_next; mem_rd <= mem_rd_next; mem_count <= mem_count_next;
            comp_wr <= comp_wr_next; comp_rd <= comp_rd_next; comp_count <= comp_count_next;
            mem_queue_count <= {{(8-(PTR_W+1)){1'b0}}, mem_count_next};
            compute_queue_count <= {{(8-(PTR_W+1)){1'b0}}, comp_count_next};
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (Q_DEPTH > 1) else $fatal("ooo_micro_sequencer Q_DEPTH must be > 1");
    end
    property p_dual_issue_possible_when_both_ready;
        @(posedge clk) disable iff (!rst_n)
            (!dma_busy && !array_busy && (mem_count != '0) && (comp_count != '0)) |=>
            (trigger_dma || shift_w_en || swap_weights) && (trigger_array || clear_ps_base);
    endproperty
    assert property (p_dual_issue_possible_when_both_ready)
        else $error("dual issue opportunity did not produce both memory and compute issue side effects");
`endif
endmodule
module systolic_pe #(
    parameter int ACT_W = 16,
    parameter int WT_W  = 16,
    parameter int PS_W  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic ce,
    input  logic sleep,
    input  logic cfg_bypass,
    input  logic cfg_dataflow,
    input  logic [3:0] cfg_mode,
    input  logic cfg_mx_native_accum,
    input  logic cfg_mx_finalize,
    input  logic [7:0] shared_exp,
    input  logic cfg_quant_en,
    input  logic [1:0] cfg_quant_scale_mode,
    input  logic [15:0] quant_scale_q8_8,
    input  logic [31:0] quant_scale_fp32,
    input  logic signed [31:0] quant_bias_i32,
    input  logic signed [15:0] act_zero_point,
    input  logic signed [15:0] wt_zero_point,
    input  logic shift_w_en,
    input  logic swap_weights,
    input  logic clear_ps,
    input  logic [WT_W-1:0] weight_in,
    input  logic [3:0] sparse_meta_in,
    input  logic [ACT_W-1:0] activation_in,
    input  logic [PS_W-1:0] partial_sum_in,
    input  logic valid_in,
    output logic [WT_W-1:0] weight_out,
    output logic [3:0] sparse_meta_out,
    output logic [ACT_W-1:0] activation_out,
    output logic [PS_W-1:0] partial_sum_out,
    output logic valid_out
);
    logic [WT_W-1:0] weight_shadow, weight_active;
    logic [3:0] sparse_meta_shadow, sparse_meta_active;
    logic [PS_W-1:0] os_accum;
    logic [ACT_W-1:0] act_q1, act_q2;
    logic [PS_W-1:0] ps_q1, ps_q2;
    logic val_q1, val_q2, clr_q1, clr_q2;
    logic pe_active_req;
    logic [31:0] mac_c_in, omni_mac_out;

    assign pe_active_req = valid_in | val_q1 | val_q2 | shift_w_en | swap_weights | clear_ps | clr_q1 | clr_q2;
    assign mac_c_in = cfg_dataflow ? ((val_q2 && !clr_q2) ? omni_mac_out : os_accum) : partial_sum_in;

    unified_fracturable_mac u_omni_mac (
        .clk(clk),
        .rst_n(rst_n),
        .ce(ce && pe_active_req && !sleep),
        .cfg_mode(cfg_mode),
        .cfg_mx_native_accum(cfg_mx_native_accum),
        .cfg_mx_finalize(cfg_mx_finalize),
        .shared_exp(shared_exp),
        .cfg_quant_en(cfg_quant_en),
        .cfg_quant_scale_mode(cfg_quant_scale_mode),
        .quant_scale_q8_8(quant_scale_q8_8),
        .quant_scale_fp32(quant_scale_fp32),
        .quant_bias_i32(quant_bias_i32),
        .act_zero_point(act_zero_point),
        .wt_zero_point(wt_zero_point),
        .a_in(weight_active),
        .b_in(activation_in),
        .sparse_meta(sparse_meta_active),
        .c_accum(mac_c_in),
        .mac_out(omni_mac_out)
    );

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            weight_shadow      <= '0;
            weight_active      <= '0;
            sparse_meta_shadow <= '0;
            sparse_meta_active <= '0;
            os_accum           <= '0;
            act_q1             <= '0;
            act_q2             <= '0;
            ps_q1              <= '0;
            ps_q2              <= '0;
            val_q1             <= 1'b0;
            val_q2             <= 1'b0;
            clr_q1             <= 1'b0;
            clr_q2             <= 1'b0;
            weight_out         <= '0;
            sparse_meta_out    <= '0;
            activation_out     <= '0;
            partial_sum_out    <= '0;
            valid_out          <= 1'b0;
        end else if (sleep) begin
            activation_out  <= '0;
            partial_sum_out <= partial_sum_in;
            valid_out       <= 1'b0;
        end else if (ce && pe_active_req) begin
            if (shift_w_en) begin
                weight_shadow      <= weight_in;
                sparse_meta_shadow <= sparse_meta_in;
            end
            if (swap_weights) begin
                weight_active      <= weight_shadow;
                sparse_meta_active <= sparse_meta_shadow;
            end

            // During sparse/weight-load shifts, pass both payload and metadata
            // downward so a multi-cycle shift_w_en stream fills the full column.
            weight_out      <= shift_w_en ? weight_in : (cfg_dataflow ? weight_in : weight_shadow);
            sparse_meta_out <= shift_w_en ? sparse_meta_in : sparse_meta_shadow;

            act_q1 <= activation_in;
            act_q2 <= act_q1;
            ps_q1  <= partial_sum_in;
            ps_q2  <= ps_q1;
            val_q1 <= valid_in;
            val_q2 <= val_q1;
            clr_q1 <= clear_ps;
            clr_q2 <= clr_q1;

            activation_out <= act_q2;
            valid_out      <= val_q2;

            if (clr_q2) begin
                partial_sum_out <= '0;
                os_accum        <= '0;
            end else if (cfg_dataflow == 1'b0) begin
                if (cfg_bypass) partial_sum_out <= ps_q2;
                else if (val_q2) partial_sum_out <= omni_mac_out;
            end else begin
                if (val_q2) os_accum <= omni_mac_out;
                partial_sum_out <= ps_q2;
            end
        end else if (ce) begin
            valid_out <= 1'b0;
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (ACT_W == 16) else $fatal("systolic_pe currently requires ACT_W=16");
        assert (WT_W  == 16) else $fatal("systolic_pe currently requires WT_W=16");
        assert (PS_W  == 32) else $fatal("systolic_pe currently requires PS_W=32");
    end
    property p_mx_finalize_requires_native;
        @(posedge clk) disable iff (!rst_n)
            cfg_mx_finalize |-> cfg_mx_native_accum;
    endproperty
    assert property (p_mx_finalize_requires_native)
        else $error("cfg_mx_finalize asserted without native MX accumulation enabled");
    initial begin
    end
`endif
endmodule

// -----------------------------------------------------------------------------
// Modular Systolic Array.
// v19 adds GQA/MHA V routing, separate sparse metadata, and a pipelined FlashAttention front-end.
// VPUs emit numerator/denominator pairs to a shared pipelined normalizer.
// -----------------------------------------------------------------------------
module systolic_array #(
    parameter int ROWS = 4,
    parameter int COLS = 4,
    parameter int ACT_W = 16,
    parameter int WT_W  = 16,
    parameter int PS_W  = 32
)(
    input  logic clk,
    input  logic rst_n,
    input  logic row_ce [0:ROWS-1],
    input  logic col_ce [0:COLS-1],
    input  logic cfg_bypass,
    input  logic cfg_dataflow,
    input  logic [3:0] cfg_mode,
    input  logic cfg_mx_native_accum,
    input  logic cfg_mx_finalize,
    input  logic [2:0] cfg_vpu_mode,
    input  logic [3:0] cfg_gqa_group_log2,
    input  logic [7:0] shared_exp,
    input  logic cfg_quant_en,
    input  logic [1:0] cfg_quant_scale_mode,
    input  logic cfg_quant_per_channel,
    input  logic [15:0] quant_scale_tensor_q8_8,
    input  logic [31:0] quant_scale_tensor_fp32,
    input  logic signed [31:0] quant_bias_tensor_i32,
    input  logic signed [15:0] act_zero_point,
    input  logic signed [15:0] wt_zero_point,
    input  logic [COLS*16-1:0] quant_scale_col_q8_8_flat,
    input  logic [COLS*32-1:0] quant_scale_col_fp32_flat,
    input  logic [COLS*32-1:0] quant_bias_col_i32_flat,
    input  logic [15:0] seq_i_base,
    input  logic [15:0] seq_j_base,
    input  logic [ROWS-1:0] row_sleep,
    input  logic shift_w_en,
    input  logic swap_weights,
    input  logic clear_ps [0:ROWS-1],
    input  logic [ACT_W-1:0] activation_in [0:ROWS-1],
    input  logic valid_in [0:ROWS-1],
    input  logic [WT_W-1:0] weight_top_in [0:COLS-1],
    input  logic [3:0] sparse_meta_top_in [0:COLS-1],
    input  logic [PS_W-1:0] ps_north_in [0:COLS-1],
    input  logic [PS_W-1:0] v_top_in [0:COLS-1],
    output logic [PS_W-1:0] partial_sum_out [0:COLS-1],
    output logic valid_out [0:COLS-1],
    output logic [ACT_W-1:0] cascade_act_out [0:ROWS-1],
    output logic cascade_val_out [0:ROWS-1]
);
    logic [ACT_W-1:0] act_right [0:ROWS-1][0:COLS];
    logic val_right [0:ROWS-1][0:COLS];
    logic [PS_W-1:0] ps_down [0:ROWS][0:COLS-1];
    logic [WT_W-1:0] wt_down [0:ROWS][0:COLS-1];
    logic [3:0] meta_down [0:ROWS][0:COLS-1];
    logic [PS_W-1:0] vpu_daisy [0:COLS];

    logic [PS_W-1:0] norm_num [0:COLS-1];
    logic [PS_W-1:0] norm_den [0:COLS-1];
    logic norm_req_valid [0:COLS-1];
    logic [PS_W-1:0] v_gqa_comb [0:COLS-1];
    logic [PS_W-1:0] v_gqa_q [0:COLS-1];
    logic [15:0] quant_scale_col_sel [0:COLS-1];
    logic [31:0] quant_scale_fp32_col_sel [0:COLS-1];
    logic signed [31:0] quant_bias_col_sel [0:COLS-1];

    assign vpu_daisy[0] = '0;

    function automatic int unsigned gqa_base_idx(input int unsigned col, input logic [3:0] group_log2);
        int unsigned group_size;
        begin
            group_size = 1 << group_log2;
            if (group_size == 0) group_size = 1;
            gqa_base_idx = (col / group_size) * group_size;
            if (gqa_base_idx >= COLS) gqa_base_idx = COLS-1;
        end
    endfunction

    always_comb begin
        for (int cc = 0; cc < COLS; cc++) begin
            v_gqa_comb[cc] = v_top_in[gqa_base_idx(cc, cfg_gqa_group_log2)];
        end
    end

    // Register the GQA broadcast result before the VPU. Without this layer, a
    // high GQA ratio can make one V-vector bit drive many VPU mux inputs
    // combinationally in large arrays.
    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int cc = 0; cc < COLS; cc++) v_gqa_q[cc] <= '0;
        end else begin
            for (int cc = 0; cc < COLS; cc++) begin
                if (col_ce[cc]) v_gqa_q[cc] <= v_gqa_comb[cc];
            end
        end
    end

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : gen_left
            assign act_right[r][0]    = activation_in[r];
            assign val_right[r][0]    = valid_in[r];
            assign cascade_act_out[r] = act_right[r][COLS];
            assign cascade_val_out[r] = val_right[r][COLS];
        end

        for (c = 0; c < COLS; c++) begin : gen_top
            assign ps_down[0][c]  = ps_north_in[c];
            assign wt_down[0][c]  = weight_top_in[c];
            assign meta_down[0][c] = sparse_meta_top_in[c];
            assign quant_scale_col_sel[c] = cfg_quant_per_channel ? quant_scale_col_q8_8_flat[(c*16) +: 16] : quant_scale_tensor_q8_8;
            assign quant_scale_fp32_col_sel[c] = cfg_quant_per_channel ? quant_scale_col_fp32_flat[(c*32) +: 32] : quant_scale_tensor_fp32;
            assign quant_bias_col_sel[c]  = cfg_quant_per_channel ? quant_bias_col_i32_flat[(c*32) +: 32] : quant_bias_tensor_i32;
        end

        for (r = 0; r < ROWS; r++) begin : gen_rows
            for (c = 0; c < COLS; c++) begin : gen_cols
                systolic_pe #(.ACT_W(ACT_W), .WT_W(WT_W), .PS_W(PS_W)) u_pe (
                    .clk(clk),
                    .rst_n(rst_n),
                    .ce(row_ce[r]),
                    .sleep(row_sleep[r]),
                    .cfg_bypass(cfg_bypass),
                    .cfg_dataflow(cfg_dataflow),
                    .cfg_mode(cfg_mode),
                    .cfg_mx_native_accum(cfg_mx_native_accum),
                    .cfg_mx_finalize(cfg_mx_finalize),
                    .shared_exp(shared_exp),
                    .cfg_quant_en(cfg_quant_en),
                    .cfg_quant_scale_mode(cfg_quant_scale_mode),
                    .quant_scale_q8_8(quant_scale_col_sel[c]),
                    .quant_scale_fp32(quant_scale_fp32_col_sel[c]),
                    .quant_bias_i32(quant_bias_col_sel[c]),
                    .act_zero_point(act_zero_point),
                    .wt_zero_point(wt_zero_point),
                    .shift_w_en(shift_w_en),
                    .swap_weights(swap_weights),
                    .clear_ps(clear_ps[r]),
                    .weight_in(wt_down[r][c]),
                    .sparse_meta_in(meta_down[r][c]),
                    .activation_in(act_right[r][c]),
                    .partial_sum_in(ps_down[r][c]),
                    .valid_in(val_right[r][c]),
                    .weight_out(wt_down[r+1][c]),
                    .sparse_meta_out(meta_down[r+1][c]),
                    .activation_out(act_right[r][c+1]),
                    .partial_sum_out(ps_down[r+1][c]),
                    .valid_out(val_right[r][c+1])
                );
            end
        end

        for (c = 0; c < COLS; c++) begin : gen_bottom
            localparam logic [15:0] COL_SEQ_OFFSET = c;
            flash_attention_vpu #(.PS_W(PS_W)) u_vpu (
                .clk(clk),
                .rst_n(rst_n),
                .ce(col_ce[c]),
                .clear_state(clear_ps[0]),
                .cfg_vpu_mode(cfg_vpu_mode),
                .seq_i(seq_i_base),
                .seq_j(seq_j_base + COL_SEQ_OFFSET),
                .x_in(ps_down[ROWS][c]),
                .x_valid(val_right[ROWS-1][c+1]),
                .v_in(v_gqa_q[c]),
                .daisy_chain_in(vpu_daisy[c]),
                .daisy_chain_out(vpu_daisy[c+1]),
                .norm_num_out(norm_num[c]),
                .norm_den_out(norm_den[c]),
                .norm_valid_out(norm_req_valid[c])
            );
        end
    endgenerate

    // Shared normalizer is the reciprocal/multiply Stage 4 backend for all columns.
    flash_shared_normalizer #(.COLS(COLS), .PS_W(PS_W)) u_shared_norm (
        .clk(clk),
        .rst_n(rst_n),
        .ce(col_ce[0]),
        .clear(clear_ps[0]),
        .num_in(norm_num),
        .den_in(norm_den),
        .valid_in(norm_req_valid),
        .norm_out(partial_sum_out),
        .valid_out(valid_out)
    );

`ifndef SYNTHESIS
    initial begin
        assert (COLS > 0) else $fatal("systolic_array COLS must be positive");
        assert (PS_W == 32) else $fatal("systolic_array currently requires PS_W=32");
    end
    property p_gqa_ratio_not_larger_than_array;
        @(posedge clk) disable iff (!rst_n)
            ((32'd1 << cfg_gqa_group_log2) <= COLS);
    endproperty
    assert property (p_gqa_ratio_not_larger_than_array)
        else $error("cfg_gqa_group_log2 selects a group larger than COLS");
`endif
endmodule


// -----------------------------------------------------------------------------
// Virtual-Channel 2D Torus Flit Router
// -----------------------------------------------------------------------------
// Compact deterministic router for v20 integration. Flit format is intentionally
// simple and compiler-friendly:
//   [FLIT_W-1]        valid/packet marker retained as payload bit by fabric users
//   [FLIT_W-2 -: 4]   destination X
//   [FLIT_W-6 -: 4]   destination Y
//   [FLIT_W-10 -: 2]  virtual channel id
// Remaining bits are payload. The router provides one-flit elastic storage per
// ingress and routes local/west/east/north/south inputs to one of the five
// outputs with fixed-priority arbitration. VC id is carried in the flit so the
// downstream fabric can allocate independent credits and avoid protocol deadlock.
// -----------------------------------------------------------------------------
// -----------------------------------------------------------------------------
// Credit-Based Virtual-Channel 2D Torus Flit Router v20.1
// -----------------------------------------------------------------------------
// Flit format:
//   [FLIT_W-2 -: 4]   destination X
//   [FLIT_W-6 -: 4]   destination Y
//   [FLIT_W-10 -: 2]  virtual channel id, valid when VC_COUNT <= 4
//
// v20 carried the VC id only as metadata and used fixed-priority arbitration.
// v20.1 gives every ingress port a physically separate one-flit queue per VC,
// eliminating HoL blocking between VCs at the input.  Each output uses a
// stateful round-robin arbiter across {input-port, VC} candidates.  A small
// per-output/per-VC credit counter models downstream credit-based flow control;
// a flit may launch only if the selected output VC has credit.
//
// The legacy m_*_ready inputs are intentionally retained for bring-up: a high
// ready pulse replenishes one credit for the output VC selected by the next
// departing flit.  Real fabrics should connect the credit inputs to explicit
// credit-return wires from downstream VC FIFOs; this wrapper avoids long
// combinational ready paths while preserving backward-compatible port names.
// -----------------------------------------------------------------------------
module vc_flit_router_2d #(
    parameter int FLIT_W = 128,
    parameter int VC_COUNT = 4,
    parameter int X_ID = 0,
    parameter int Y_ID = 0,
    parameter int CREDIT_DEPTH = 4
)(
    input  logic clk,
    input  logic rst_n,
    input  logic [FLIT_W-1:0] s_local_data, input logic s_local_valid, output logic s_local_ready,
    input  logic [FLIT_W-1:0] s_west_data,  input logic s_west_valid,  output logic s_west_ready,
    input  logic [FLIT_W-1:0] s_east_data,  input logic s_east_valid,  output logic s_east_ready,
    input  logic [FLIT_W-1:0] s_north_data, input logic s_north_valid, output logic s_north_ready,
    input  logic [FLIT_W-1:0] s_south_data, input logic s_south_valid, output logic s_south_ready,
    output logic [FLIT_W-1:0] m_local_data, output logic m_local_valid, input logic m_local_ready, input logic [VC_COUNT-1:0] m_local_credit_return,
    output logic [FLIT_W-1:0] m_west_data,  output logic m_west_valid,  input logic m_west_ready,  input logic [VC_COUNT-1:0] m_west_credit_return,
    output logic [FLIT_W-1:0] m_east_data,  output logic m_east_valid,  input logic m_east_ready,  input logic [VC_COUNT-1:0] m_east_credit_return,
    output logic [FLIT_W-1:0] m_north_data, output logic m_north_valid, input logic m_north_ready, input logic [VC_COUNT-1:0] m_north_credit_return,
    output logic [FLIT_W-1:0] m_south_data, output logic m_south_valid, input logic m_south_ready, input logic [VC_COUNT-1:0] m_south_credit_return
);
    typedef enum logic [2:0] {P_LOCAL=3'd0, P_WEST=3'd1, P_EAST=3'd2, P_NORTH=3'd3, P_SOUTH=3'd4} port_t;
    localparam int PORTS = 5;
    localparam int CAND  = PORTS * VC_COUNT;
    localparam int RR_W  = (CAND <= 1) ? 1 : $clog2(CAND);
    localparam int CREDIT_W = (CREDIT_DEPTH <= 1) ? 1 : $clog2(CREDIT_DEPTH+1);
    localparam logic [3:0] X_ID4 = X_ID;
    localparam logic [3:0] Y_ID4 = Y_ID;
    localparam logic [CREDIT_W-1:0] CREDIT_MAX = CREDIT_DEPTH;

    logic [FLIT_W-1:0] in_data [0:PORTS-1];
    logic in_valid [0:PORTS-1];
    logic in_ready [0:PORTS-1];
    logic out_ready_compat [0:PORTS-1];
    logic [VC_COUNT-1:0] out_credit_return [0:PORTS-1];
    logic [FLIT_W-1:0] out_data [0:PORTS-1];
    logic out_valid [0:PORTS-1];

    logic [FLIT_W-1:0] vc_q_data [0:PORTS-1][0:VC_COUNT-1];
    logic vc_q_valid [0:PORTS-1][0:VC_COUNT-1];
    logic [CREDIT_W-1:0] credit [0:PORTS-1][0:VC_COUNT-1];

    // v20.2: one net credit update per output/VC per cycle. Downstream returns
    // credits explicitly per VC via m_*_credit_return; m_*_ready only controls
    // compatibility/output observation and does not mint credits. credit_return
    // and credit_consume are folded into one saturated assignment, eliminating
    // the v20.1 "last nonblocking assignment wins" credit leak.
    logic credit_return [0:PORTS-1][0:VC_COUNT-1];
    logic credit_consume [0:PORTS-1][0:VC_COUNT-1];

    logic [RR_W-1:0] rr_ptr [0:PORTS-1];
    logic sel_found [0:PORTS-1];
    logic [RR_W-1:0] sel_flat [0:PORTS-1];

    assign in_data[0]=s_local_data; assign in_valid[0]=s_local_valid; assign s_local_ready=in_ready[0];
    assign in_data[1]=s_west_data;  assign in_valid[1]=s_west_valid;  assign s_west_ready=in_ready[1];
    assign in_data[2]=s_east_data;  assign in_valid[2]=s_east_valid;  assign s_east_ready=in_ready[2];
    assign in_data[3]=s_north_data; assign in_valid[3]=s_north_valid; assign s_north_ready=in_ready[3];
    assign in_data[4]=s_south_data; assign in_valid[4]=s_south_valid; assign s_south_ready=in_ready[4];

    assign m_local_data=out_data[0]; assign m_local_valid=out_valid[0]; assign out_ready_compat[0]=m_local_ready;
    assign m_west_data=out_data[1];  assign m_west_valid=out_valid[1];  assign out_ready_compat[1]=m_west_ready;
    assign m_east_data=out_data[2];  assign m_east_valid=out_valid[2];  assign out_ready_compat[2]=m_east_ready;
    assign m_north_data=out_data[3]; assign m_north_valid=out_valid[3]; assign out_ready_compat[3]=m_north_ready;
    assign m_south_data=out_data[4]; assign m_south_valid=out_valid[4]; assign out_ready_compat[4]=m_south_ready;

    assign out_credit_return[0] = m_local_credit_return;
    assign out_credit_return[1] = m_west_credit_return;
    assign out_credit_return[2] = m_east_credit_return;
    assign out_credit_return[3] = m_north_credit_return;
    assign out_credit_return[4] = m_south_credit_return;

    function automatic int unsigned vc_of(input logic [FLIT_W-1:0] flit);
        int unsigned raw_vc;
        begin
            raw_vc = flit[FLIT_W-10 -: 2];
            vc_of = (raw_vc >= VC_COUNT) ? (VC_COUNT-1) : raw_vc;
        end
    endfunction

    function automatic port_t route_for(input logic [FLIT_W-1:0] flit);
        logic [3:0] dx, dy;
        begin
            dx = flit[FLIT_W-2 -: 4];
            dy = flit[FLIT_W-6 -: 4];
            if ((dx == X_ID4) && (dy == Y_ID4)) route_for = P_LOCAL;
            else if (dx != X_ID4) route_for = (dx > X_ID4) ? P_EAST : P_WEST;
            else route_for = (dy > Y_ID4) ? P_NORTH : P_SOUTH;
        end
    endfunction

    function automatic int unsigned flat_port(input int unsigned flat);
        flat_port = flat / VC_COUNT;
    endfunction

    function automatic int unsigned flat_vc(input int unsigned flat);
        flat_vc = flat % VC_COUNT;
    endfunction

    function automatic logic [RR_W-1:0] rr_inc(input logic [RR_W-1:0] p);
        rr_inc = (p == CAND-1) ? '0 : p + 1'b1;
    endfunction

    function automatic logic [CREDIT_W-1:0] credit_net_next(
        input logic [CREDIT_W-1:0] cur,
        input logic ret,
        input logic cons
    );
        logic [CREDIT_W:0] tmp;
        begin
            tmp = {1'b0, cur};
            if (ret && (cur != CREDIT_MAX)) tmp = tmp + 1'b1;
            if (cons && (tmp != '0))       tmp = tmp - 1'b1;
            if (tmp[CREDIT_W-1:0] > CREDIT_MAX) credit_net_next = CREDIT_MAX;
            else credit_net_next = tmp[CREDIT_W-1:0];
        end
    endfunction

    always_comb begin
        for (int p=0; p<PORTS; p++) begin
            int unsigned v;
            v = vc_of(in_data[p]);
            in_ready[p] = !vc_q_valid[p][v];
        end

        for (int o=0; o<PORTS; o++) begin
            sel_found[o] = 1'b0;
            sel_flat[o]  = rr_ptr[o];
            for (int step=0; step<CAND; step++) begin
                int unsigned flat;
                int unsigned ip;
                int unsigned ivc;
                flat = (rr_ptr[o] + step) % CAND;
                ip   = flat_port(flat);
                ivc  = flat_vc(flat);
                if (!sel_found[o] &&
                    vc_q_valid[ip][ivc] &&
                    (route_for(vc_q_data[ip][ivc]) == port_t'(o)) &&
                    (credit[o][ivc] != '0)) begin
                    sel_found[o] = 1'b1;
                    sel_flat[o]  = flat[RR_W-1:0];
                end
            end
        end

        for (int p=0; p<PORTS; p++) begin
            for (int v=0; v<VC_COUNT; v++) begin
                credit_return[p][v]  = out_credit_return[p][v] && (credit[p][v] != CREDIT_MAX);
                credit_consume[p][v] = 1'b0;
            end
        end
        for (int o=0; o<PORTS; o++) begin
            if (sel_found[o]) begin
                credit_consume[o][flat_vc(sel_flat[o])] = 1'b1;
            end
        end
    end

    always_ff @(posedge clk) begin
        if (!rst_n) begin
            for (int p=0; p<PORTS; p++) begin
                out_data[p] <= '0;
                out_valid[p] <= 1'b0;
                rr_ptr[p] <= '0;
                for (int v=0; v<VC_COUNT; v++) begin
                    vc_q_data[p][v] <= '0;
                    vc_q_valid[p][v] <= 1'b0;
                    credit[p][v] <= CREDIT_MAX;
                end
            end
        end else begin
            for (int p=0; p<PORTS; p++) begin
                out_valid[p] <= 1'b0;
            end

            // Capture incoming flits into per-input, per-VC queues. The queues
            // are deliberately one flit deep in this baseline; scaling to deeper
            // FIFOs keeps the same credit/arbitration interface.
            for (int p=0; p<PORTS; p++) begin
                int unsigned v;
                v = vc_of(in_data[p]);
                if (in_valid[p] && in_ready[p]) begin
                    vc_q_data[p][v] <= in_data[p];
                    vc_q_valid[p][v] <= 1'b1;
                end
            end

            // Per-output round-robin launch.  The launch only marks a VC queue
            // empty; credit is not modified here. Credit is updated once below.
            for (int o=0; o<PORTS; o++) begin
                if (sel_found[o]) begin
                    int unsigned ip;
                    int unsigned ivc;
                    ip  = flat_port(sel_flat[o]);
                    ivc = flat_vc(sel_flat[o]);
                    out_data[o] <= vc_q_data[ip][ivc];
                    out_valid[o] <= 1'b1;
                    vc_q_valid[ip][ivc] <= 1'b0;
                    rr_ptr[o] <= rr_inc(sel_flat[o]);
                end
            end

            for (int p=0; p<PORTS; p++) begin
                for (int v=0; v<VC_COUNT; v++) begin
                    credit[p][v] <= credit_net_next(credit[p][v], credit_return[p][v], credit_consume[p][v]);
                end
            end
        end
    end

`ifndef SYNTHESIS
    initial begin
        assert (FLIT_W >= 32) else $fatal("vc_flit_router_2d FLIT_W must be at least 32");
        assert (VC_COUNT > 0) else $fatal("VC_COUNT must be positive");
        assert (VC_COUNT <= 4) else $fatal("This compact flit format exposes only two VC bits");
        assert (CREDIT_DEPTH > 0) else $fatal("CREDIT_DEPTH must be positive");
    end
    property p_input_accepts_only_free_vc;
        @(posedge clk) disable iff (!rst_n)
            (s_local_valid && s_local_ready) |-> !vc_q_valid[0][vc_of(s_local_data)];
    endproperty
    assert property (p_input_accepts_only_free_vc)
        else $error("router accepted into an occupied VC queue");

    property p_credit_bounds;
        @(posedge clk) disable iff (!rst_n)
            (credit[0][0] <= CREDIT_MAX);
    endproperty
    assert property (p_credit_bounds)
        else $error("router credit exceeded CREDIT_MAX");
`endif
endmodule

module hyperion_exascale_node #(
    parameter int ROWS        = 4,
    parameter int COLS        = 4,
    parameter int S_AXIS_W    = 64,
    parameter int M_AXIS_W    = COLS * 32,
    parameter int WT_TOP_W    = COLS * 16,
    parameter int META_AXIS_W = COLS * 4,
    parameter int TCSM_DEPTH  = 256,
    parameter int TCSM_AW     = (TCSM_DEPTH <= 1) ? 1 : $clog2(TCSM_DEPTH)
)(
    input  logic clk,
    input  logic rst_n,

    input  logic [31:0] ir_in,
    input  logic ir_valid,
    input  logic [3:0] cfg_mode,
    input  logic cfg_mx_native_accum,
    input  logic cfg_mx_finalize,
    input  logic [2:0] cfg_vpu_mode,
    input  logic [3:0] cfg_gqa_group_log2,
    input  logic cfg_bypass,
    input  logic cfg_dataflow,
    input  logic cfg_allreduce,
    input  logic cfg_allreduce_fp, // south collective can use reference FP32 add; east 16-bit lanes remain quantized/raw payload adds
    input  logic cfg_broadcast,
    input  logic cfg_rope_en,
    input  logic [7:0] shared_exp,
    input  logic [15:0] seq_i_base,
    input  logic [15:0] seq_j_base,
    input  logic [ROWS-1:0] row_sleep,
    input  logic dma_busy,
    input  logic array_busy,

    // v21 quantization configuration. Scale is Q8.8 for integer/quantized MAC modes;
    // bias is INT32. Per-channel mode selects column-indexed scale/bias.
    input  logic cfg_quant_en,
    input  logic [1:0] cfg_quant_scale_mode,
    input  logic cfg_quant_per_channel,
    input  logic [15:0] quant_scale_tensor_q8_8,
    input  logic [31:0] quant_scale_tensor_fp32,
    input  logic signed [31:0] quant_bias_tensor_i32,
    input  logic signed [15:0] act_zero_point,
    input  logic signed [15:0] wt_zero_point,
    input  logic [COLS*16-1:0] quant_scale_col_q8_8_flat,
    input  logic [COLS*32-1:0] quant_scale_col_fp32_flat,
    input  logic [COLS*32-1:0] quant_bias_col_i32_flat,

    // v21 TMA-lite descriptor and payload stream. The TMA can autonomously write
    // weights/V/metadata TCSMs. Legacy direct load ports below remain available.
    input  logic tma_desc_valid,
    output logic tma_desc_ready,
    input  logic [63:0] tma_desc_base_addr,
    input  logic [15:0] tma_desc_dim_m,
    input  logic [15:0] tma_desc_dim_n,
    input  logic [15:0] tma_desc_stride_m,
    input  logic [15:0] tma_desc_stride_n,
    input  logic [15:0] tma_desc_tile_m,
    input  logic [15:0] tma_desc_tile_n,
    input  logic [1:0]  tma_desc_dst_kind,
    input  logic tma_desc_dst_bank,
    input  logic [M_AXIS_W-1:0] tma_stream_data,
    input  logic tma_stream_valid,
    output logic tma_stream_ready,
    output logic tma_busy,
    output logic tma_done,

    // v21 PagedAttention scaffold. Lookup translates virtual KV pages into
    // physical page numbers; later TMA revisions can consume this response.
    input  logic kv_lookup_valid,
    output logic kv_lookup_ready,
    input  logic [11:0] kv_lookup_vpn,
    output logic kv_lookup_resp_valid,
    output logic kv_lookup_miss,
    output logic [23:0] kv_lookup_ppn,
    output logic kv_pager_stall,
    output logic kv_fault_valid,
    output logic [11:0] kv_fault_vpn,
    input  logic kv_fault_clear,
    input  logic kv_ptw_write_valid,
    input  logic [7:0] kv_ptw_write_index,
    input  logic [11:0] kv_ptw_write_vpn,
    input  logic [23:0] kv_ptw_write_ppn,
    input  logic kv_ptw_write_valid_bit,

    // TCSM load/read interface. weight_top_flat and v_top_flat are load payloads
    // into local ping-pong memories, not direct every-cycle core inputs. Weight
    // and V memories have independent read addresses so later tiling can control
    // K/V movement separately.
    input  logic [WT_TOP_W-1:0] weight_top_flat,
    input  logic weight_load_valid,
    input  logic weight_load_bank,
    input  logic [TCSM_AW-1:0] weight_load_addr,
    input  logic [M_AXIS_W-1:0] v_top_flat,
    input  logic v_load_valid,
    input  logic v_load_bank,
    input  logic [TCSM_AW-1:0] v_load_addr,
    input  logic [TCSM_AW-1:0] weight_read_addr,
    input  logic [TCSM_AW-1:0] v_read_addr,
    input  logic tcsm_swap,

    input  logic [S_AXIS_W-1:0] s_axis_west_tdata,
    input  logic s_axis_west_tvalid,
    output logic s_axis_west_tready,
    output logic [S_AXIS_W-1:0] m_axis_east_tdata,
    output logic m_axis_east_tvalid,
    input  logic m_axis_east_tready,

    input  logic [M_AXIS_W-1:0] s_axis_north_tdata,
    input  logic s_axis_north_tvalid,
    output logic s_axis_north_tready,
    output logic [M_AXIS_W-1:0] m_axis_south_tdata,
    output logic m_axis_south_tvalid,
    input  logic m_axis_south_tready,

    // Separate sparse metadata AXI-style stream. Each column receives one
    // 4-bit metadata bundle per shifted sparse-weight payload: {idx1, idx0}.
    input  logic [META_AXIS_W-1:0] s_axis_meta_tdata,
    input  logic s_axis_meta_tvalid,
    output logic s_axis_meta_tready
);
    localparam int EAST_ALIGN_LAT  = 2 * COLS;
    localparam int SOUTH_ALIGN_LAT = (2 * ROWS) + 5; // PE + 4-stage VPU + normalizer request latency approximation

    logic [S_AXIS_W-1:0] rope_tdata;
    logic rope_tvalid, rope_tready;

    rope_engine #(.DATA_W(S_AXIS_W)) u_rope (
        .clk(clk),
        .rst_n(rst_n),
        .cfg_rope_en(cfg_rope_en),
        .s_tdata(s_axis_west_tdata),
        .s_tvalid(s_axis_west_tvalid),
        .s_tready(s_axis_west_tready),
        .m_tdata(rope_tdata),
        .m_tvalid(rope_tvalid),
        .m_tready(rope_tready)
    );

    logic shift_w_en, swap_weights, clear_ps_base;
    logic trigger_dma, trigger_array;
    ooo_micro_sequencer u_seq (
        .clk(clk),
        .rst_n(rst_n),
        .ir_in(ir_in),
        .ir_valid(ir_valid),
        .shift_w_en(shift_w_en),
        .swap_weights(swap_weights),
        .clear_ps_base(clear_ps_base),
        .dma_busy(dma_busy || kv_pager_stall),
        .array_busy(array_busy || kv_pager_stall),
        .trigger_dma(trigger_dma),
        .trigger_array(trigger_array),
        .mem_issue_valid(),
        .compute_issue_valid(),
        .dual_issue_valid(),
        .mem_queue_count(),
        .compute_queue_count()
    );

    logic rx_valid, rx_pop, rx_full, rx_afull;
    logic [S_AXIS_W-1:0] rx_tdata;
    sync_fifo #(.DATA_W(S_AXIS_W), .DEPTH(32)) u_rx_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .push(rope_tvalid && rope_tready),
        .data_in(rope_tdata),
        .pop(rx_pop),
        .data_out(rx_tdata),
        .valid_out(rx_valid),
        .empty(),
        .full(rx_full),
        .almost_full(rx_afull)
    );
    assign rope_tready = !rx_full;

    logic north_valid, north_pop, north_full, north_afull;
    logic [M_AXIS_W-1:0] north_tdata;
    sync_fifo #(.DATA_W(M_AXIS_W), .DEPTH(32)) u_north_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .push(s_axis_north_tvalid && s_axis_north_tready),
        .data_in(s_axis_north_tdata),
        .pop(north_pop),
        .data_out(north_tdata),
        .valid_out(north_valid),
        .empty(),
        .full(north_full),
        .almost_full(north_afull)
    );
    assign s_axis_north_tready = !north_full;


    logic tma_load_valid;
    logic [1:0] tma_load_dst_kind;
    logic tma_load_bank;
    logic [TCSM_AW-1:0] tma_load_addr;
    logic [63:0] tma_load_addr_full;
    logic [M_AXIS_W-1:0] tma_load_data;
    logic tma_desc_ready_int, tma_desc_error;
    logic weight_load_valid_eff, v_load_valid_eff;
    assign tma_desc_ready = tma_desc_ready_int && !kv_pager_stall;
    logic weight_load_bank_eff, v_load_bank_eff;
    logic [TCSM_AW-1:0] weight_load_addr_eff, v_load_addr_eff;
    logic [WT_TOP_W-1:0] weight_load_data_eff;
    logic [M_AXIS_W-1:0] v_load_data_eff;
    logic meta_valid, meta_pop, meta_full, meta_afull;
    logic [META_AXIS_W-1:0] meta_tdata;
    sync_fifo #(.DATA_W(META_AXIS_W), .DEPTH(32)) u_sparse_meta_fifo (
        .clk(clk),
        .rst_n(rst_n),
        .push((s_axis_meta_tvalid && s_axis_meta_tready) || (tma_load_valid && (tma_load_dst_kind == 2'd2))),
        .data_in((tma_load_valid && (tma_load_dst_kind == 2'd2)) ? tma_load_data[META_AXIS_W-1:0] : s_axis_meta_tdata),
        .pop(meta_pop),
        .data_out(meta_tdata),
        .valid_out(meta_valid),
        .empty(),
        .full(meta_full),
        .almost_full(meta_afull)
    );
    assign s_axis_meta_tready = !meta_full;

    logic east_obuf_ready, south_obuf_ready;
    logic core_step_root, ingress_ce;
    logic row_ce [0:ROWS-1];
    logic col_ce [0:COLS-1];
    logic sparse_meta_needed;
    logic sparse_meta_ready_for_step;

    assign sparse_meta_needed = (cfg_mode == 4'h8) && shift_w_en;
    assign sparse_meta_ready_for_step = !sparse_meta_needed || meta_valid;
    assign core_step_root = east_obuf_ready && south_obuf_ready && sparse_meta_ready_for_step && !kv_pager_stall;

    ce_relay_grid #(.ROWS(ROWS), .COLS(COLS)) u_ce_relay (
        .clk(clk),
        .rst_n(rst_n),
        .root_step(core_step_root),
        .ingress_ce(ingress_ce),
        .row_ce(row_ce),
        .col_ce(col_ce)
    );

    assign rx_pop    = ingress_ce && rx_valid;
    assign north_pop = ingress_ce && north_valid;
    // Drain stale/irrelevant metadata during non-sparse operation so the sideband
    // FIFO cannot backpressure unrelated dense workloads. In sparse mode, hold
    // metadata until a weight-shift pulse consumes it with the corresponding
    // sparse payload.
    assign meta_pop  = ingress_ce && meta_valid && (sparse_meta_needed || (cfg_mode != 4'h8));

    logic [15:0] activation_in [0:ROWS-1];
    logic valid_in [0:ROWS-1];
    logic clear_ps_arr [0:ROWS-1];
    logic [31:0] partial_sum_out [0:COLS-1];
    logic valid_out [0:COLS-1];
    logic [15:0] cascade_act_out [0:ROWS-1];
    logic cascade_val_out [0:ROWS-1];
    logic [31:0] ps_north_in [0:COLS-1];
    logic [15:0] weight_top_in [0:COLS-1];
    logic [3:0] sparse_meta_top_in [0:COLS-1];
    logic [31:0] v_top_in [0:COLS-1];
    logic [WT_TOP_W-1:0] weight_tcsm_bus;
    logic [M_AXIS_W-1:0] v_tcsm_bus;
    logic weight_active_bank, v_active_bank;
    logic [S_AXIS_W-1:0] east_payload;
    logic [M_AXIS_W-1:0] south_payload;

    tma_tensor_loader #(.DATA_W(M_AXIS_W), .ADDR_W(TCSM_AW)) u_tma_loader (
        .clk(clk),
        .rst_n(rst_n),
        .desc_valid(tma_desc_valid && !kv_pager_stall),
        .desc_ready(tma_desc_ready_int),
        .desc_base_addr(tma_desc_base_addr),
        .desc_dim_m(tma_desc_dim_m),
        .desc_dim_n(tma_desc_dim_n),
        .desc_stride_m(tma_desc_stride_m),
        .desc_stride_n(tma_desc_stride_n),
        .desc_tile_m(tma_desc_tile_m),
        .desc_tile_n(tma_desc_tile_n),
        .desc_dst_kind(tma_desc_dst_kind),
        .desc_dst_bank(tma_desc_dst_bank),
        .hold(kv_pager_stall),
        .stream_data(tma_stream_data),
        .stream_valid(tma_stream_valid),
        .stream_ready(tma_stream_ready),
        .load_valid(tma_load_valid),
        .load_dst_kind(tma_load_dst_kind),
        .load_bank(tma_load_bank),
        .load_addr(tma_load_addr),
        .load_addr_full(tma_load_addr_full),
        .load_data(tma_load_data),
        .busy(tma_busy),
        .done(tma_done),
        .desc_error(tma_desc_error)
    );

    kv_page_table #(.VPN_W(12), .PPN_W(24), .PAGE_COUNT(256)) u_kv_page_table (
        .clk(clk),
        .rst_n(rst_n),
        .lookup_valid(kv_lookup_valid),
        .lookup_ready(kv_lookup_ready),
        .lookup_vpn(kv_lookup_vpn),
        .lookup_resp_valid(kv_lookup_resp_valid),
        .lookup_miss(kv_lookup_miss),
        .lookup_ppn(kv_lookup_ppn),
        .pager_stall(kv_pager_stall),
        .fault_valid(kv_fault_valid),
        .fault_vpn(kv_fault_vpn),
        .fault_clear(kv_fault_clear),
        .ptw_write_valid(kv_ptw_write_valid),
        .ptw_write_index(kv_ptw_write_index),
        .ptw_write_vpn(kv_ptw_write_vpn),
        .ptw_write_ppn(kv_ptw_write_ppn),
        .ptw_write_valid_bit(kv_ptw_write_valid_bit)
    );

    assign weight_load_valid_eff = weight_load_valid || (tma_load_valid && (tma_load_dst_kind == 2'd0));
    assign weight_load_bank_eff  = (tma_load_valid && (tma_load_dst_kind == 2'd0)) ? tma_load_bank : weight_load_bank;
    assign weight_load_addr_eff  = (tma_load_valid && (tma_load_dst_kind == 2'd0)) ? tma_load_addr : weight_load_addr;
    assign weight_load_data_eff  = (tma_load_valid && (tma_load_dst_kind == 2'd0)) ? tma_load_data[WT_TOP_W-1:0] : weight_top_flat;

    assign v_load_valid_eff = v_load_valid || (tma_load_valid && (tma_load_dst_kind == 2'd1));
    assign v_load_bank_eff  = (tma_load_valid && (tma_load_dst_kind == 2'd1)) ? tma_load_bank : v_load_bank;
    assign v_load_addr_eff  = (tma_load_valid && (tma_load_dst_kind == 2'd1)) ? tma_load_addr : v_load_addr;
    assign v_load_data_eff  = (tma_load_valid && (tma_load_dst_kind == 2'd1)) ? tma_load_data : v_top_flat;

    ping_pong_vector_tcsm #(.DATA_W(WT_TOP_W), .DEPTH(TCSM_DEPTH), .ADDR_W(TCSM_AW)) u_weight_tcsm (
        .clk(clk),
        .rst_n(rst_n),
        .load_en(weight_load_valid_eff),
        .load_bank(weight_load_bank_eff),
        .load_addr(weight_load_addr_eff),
        .load_data(weight_load_data_eff),
        .read_addr(weight_read_addr),
        .swap_banks(tcsm_swap),
        .read_data(weight_tcsm_bus),
        .active_bank(weight_active_bank)
    );

    ping_pong_vector_tcsm #(.DATA_W(M_AXIS_W), .DEPTH(TCSM_DEPTH), .ADDR_W(TCSM_AW)) u_v_tcsm (
        .clk(clk),
        .rst_n(rst_n),
        .load_en(v_load_valid_eff),
        .load_bank(v_load_bank_eff),
        .load_addr(v_load_addr_eff),
        .load_data(v_load_data_eff),
        .read_addr(v_read_addr),
        .swap_banks(tcsm_swap),
        .read_data(v_tcsm_bus),
        .active_bank(v_active_bank)
    );

    genvar i;
    generate
        for (i = 0; i < ROWS; i++) begin : gen_skew_and_east_align
            logic [15:0] act_delay [0:(2*i)];
            logic val_delay [0:(2*i)];
            logic [15:0] east_addend_delay [0:EAST_ALIGN_LAT];
            logic east_addend_valid [0:EAST_ALIGN_LAT];

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    for (int d = 0; d <= (2*i); d++) begin
                        act_delay[d] <= 16'd0;
                        val_delay[d] <= 1'b0;
                    end
                    for (int d = 0; d <= EAST_ALIGN_LAT; d++) begin
                        east_addend_delay[d] <= 16'd0;
                        east_addend_valid[d] <= 1'b0;
                    end
                end else if (ingress_ce) begin
                    if (rx_pop) begin
                        act_delay[0] <= rx_tdata[(i*16) +: 16];
                        val_delay[0] <= 1'b1;
                    end else begin
                        act_delay[0] <= 16'd0;
                        val_delay[0] <= 1'b0;
                    end
                    for (int d = 1; d <= (2*i); d++) begin
                        act_delay[d] <= act_delay[d-1];
                        val_delay[d] <= val_delay[d-1];
                    end

                    east_addend_delay[0] <= cfg_broadcast ? rx_tdata[(i*16) +: 16] : act_delay[2*i];
                    east_addend_valid[0] <= cfg_broadcast ? rx_pop : val_delay[2*i];
                    for (int d = 1; d <= EAST_ALIGN_LAT; d++) begin
                        east_addend_delay[d] <= east_addend_delay[d-1];
                        east_addend_valid[d] <= east_addend_valid[d-1];
                    end
                end
            end

            assign activation_in[i] = cfg_broadcast ? rx_tdata[(i*16) +: 16] : act_delay[2*i];
            assign valid_in[i]      = cfg_broadcast ? rx_pop : val_delay[2*i];
            assign clear_ps_arr[i]  = clear_ps_base;
            // Collective path is a quantized/integer payload adder. If these
            // lanes carry FP activations in a later ISA, replace this with the
            // appropriate FP add path or convert before collectives.
            assign east_payload[(i*16) +: 16] = cfg_allreduce ?
                (cascade_act_out[i] + (east_addend_valid[EAST_ALIGN_LAT] ? east_addend_delay[EAST_ALIGN_LAT] : 16'd0)) :
                cascade_act_out[i];
        end

        if (S_AXIS_W > (ROWS*16)) begin : gen_east_pad
            assign east_payload[S_AXIS_W-1:ROWS*16] = '0;
        end

        for (i = 0; i < COLS; i++) begin : gen_cols_flatten_and_south_align
            logic [31:0] south_addend_delay [0:SOUTH_ALIGN_LAT];
            logic south_addend_valid [0:SOUTH_ALIGN_LAT];

            assign ps_north_in[i]       = north_valid ? north_tdata[(i*32) +: 32] : 32'd0;
            assign weight_top_in[i]     = weight_tcsm_bus[(i*16) +: 16];
            assign sparse_meta_top_in[i] = meta_valid ? meta_tdata[(i*4) +: 4] : 4'd0;
            assign v_top_in[i]          = v_tcsm_bus[(i*32) +: 32];

            always_ff @(posedge clk) begin
                if (!rst_n) begin
                    for (int d = 0; d <= SOUTH_ALIGN_LAT; d++) begin
                        south_addend_delay[d] <= 32'd0;
                        south_addend_valid[d] <= 1'b0;
                    end
                end else if (ingress_ce) begin
                    south_addend_delay[0] <= ps_north_in[i];
                    south_addend_valid[0] <= north_valid;
                    for (int d = 1; d <= SOUTH_ALIGN_LAT; d++) begin
                        south_addend_delay[d] <= south_addend_delay[d-1];
                        south_addend_valid[d] <= south_addend_valid[d-1];
                    end
                end
            end

            // South collectives can operate either on raw quantized/INT32 payloads
            // or on FP32 lanes. East activations remain 16-bit quantized/raw in
            // this top-level wrapper; use a packetized VC route for mixed FP
            // activation collectives.
            logic [31:0] south_addend_aligned;
            logic [31:0] south_raw_sum;
            logic [31:0] south_fp_sum;
            assign south_addend_aligned = south_addend_valid[SOUTH_ALIGN_LAT] ? south_addend_delay[SOUTH_ALIGN_LAT] : 32'd0;
            assign south_raw_sum = partial_sum_out[i] + south_addend_aligned;
            fp32_adder u_south_allreduce_fp (.a(partial_sum_out[i]), .b(south_addend_aligned), .sum(south_fp_sum));
            assign south_payload[(i*32) +: 32] = cfg_allreduce ?
                (cfg_allreduce_fp ? south_fp_sum : south_raw_sum) :
                partial_sum_out[i];
        end
    endgenerate

    logic east_core_valid, south_core_valid;
    assign east_core_valid  = |cascade_val_out;
    assign south_core_valid = |valid_out;

    axis_hold_reg #(.DATA_W(S_AXIS_W)) u_east_hold (
        .clk(clk),
        .rst_n(rst_n),
        .ce(ingress_ce),
        .s_data(east_payload),
        .s_valid(east_core_valid),
        .s_ready(east_obuf_ready),
        .m_data(m_axis_east_tdata),
        .m_valid(m_axis_east_tvalid),
        .m_ready(m_axis_east_tready)
    );

    axis_hold_reg #(.DATA_W(M_AXIS_W)) u_south_hold (
        .clk(clk),
        .rst_n(rst_n),
        .ce(ingress_ce),
        .s_data(south_payload),
        .s_valid(south_core_valid),
        .s_ready(south_obuf_ready),
        .m_data(m_axis_south_tdata),
        .m_valid(m_axis_south_tvalid),
        .m_ready(m_axis_south_tready)
    );

    systolic_array #(.ROWS(ROWS), .COLS(COLS), .ACT_W(16), .WT_W(16), .PS_W(32)) u_core (
        .clk(clk),
        .rst_n(rst_n),
        .row_ce(row_ce),
        .col_ce(col_ce),
        .cfg_bypass(cfg_bypass),
        .cfg_dataflow(cfg_dataflow),
        .cfg_mode(cfg_mode),
        .cfg_mx_native_accum(cfg_mx_native_accum),
        .cfg_mx_finalize(cfg_mx_finalize),
        .cfg_vpu_mode(cfg_vpu_mode),
        .cfg_gqa_group_log2(cfg_gqa_group_log2),
        .shared_exp(shared_exp),
        .cfg_quant_en(cfg_quant_en),
        .cfg_quant_scale_mode(cfg_quant_scale_mode),
        .cfg_quant_per_channel(cfg_quant_per_channel),
        .quant_scale_tensor_q8_8(quant_scale_tensor_q8_8),
        .quant_scale_tensor_fp32(quant_scale_tensor_fp32),
        .quant_bias_tensor_i32(quant_bias_tensor_i32),
        .act_zero_point(act_zero_point),
        .wt_zero_point(wt_zero_point),
        .quant_scale_col_q8_8_flat(quant_scale_col_q8_8_flat),
        .quant_scale_col_fp32_flat(quant_scale_col_fp32_flat),
        .quant_bias_col_i32_flat(quant_bias_col_i32_flat),
        .seq_i_base(seq_i_base),
        .seq_j_base(seq_j_base),
        .row_sleep(row_sleep),
        .shift_w_en(shift_w_en),
        .swap_weights(swap_weights),
        .clear_ps(clear_ps_arr),
        .activation_in(activation_in),
        .valid_in(valid_in),
        .weight_top_in(weight_top_in),
        .sparse_meta_top_in(sparse_meta_top_in),
        .ps_north_in(ps_north_in),
        .v_top_in(v_top_in),
        .partial_sum_out(partial_sum_out),
        .valid_out(valid_out),
        .cascade_act_out(cascade_act_out),
        .cascade_val_out(cascade_val_out)
    );

`ifndef SYNTHESIS
    initial begin
        assert (ROWS > 0) else $fatal("ROWS must be positive");
        assert (COLS > 0) else $fatal("COLS must be positive");
        assert (S_AXIS_W >= ROWS*16) else $fatal("S_AXIS_W must be >= ROWS*16");
        assert (M_AXIS_W >= COLS*32) else $fatal("M_AXIS_W must be >= COLS*32");
        assert (WT_TOP_W >= COLS*16) else $fatal("WT_TOP_W must be >= COLS*16");
        assert (M_AXIS_W >= WT_TOP_W) else $fatal("v21 TMA stream DATA_W=M_AXIS_W must be >= WT_TOP_W for weight loads");
        assert (META_AXIS_W >= COLS*4) else $fatal("META_AXIS_W must be >= COLS*4");
        assert (TCSM_DEPTH > 0) else $fatal("TCSM_DEPTH must be positive");
    end
    property p_sparse_shift_has_metadata;
        @(posedge clk) disable iff (!rst_n)
            ((cfg_mode == 4'h8) && shift_w_en && ingress_ce) |-> meta_valid;
    endproperty
    assert property (p_sparse_shift_has_metadata)
        else $error("Sparse weight shift advanced without metadata");
    property p_pager_stall_blocks_tma_accept;
        @(posedge clk) disable iff (!rst_n) kv_pager_stall |-> !tma_desc_ready;
    endproperty
    assert property (p_pager_stall_blocks_tma_accept)
        else $error("TMA descriptor accepted during unresolved KV pager fault");

    property p_quant_mode_supported;
        @(posedge clk) disable iff (!rst_n) cfg_quant_scale_mode <= 2'd2;
    endproperty
    assert property (p_quant_mode_supported)
        else $error("unsupported cfg_quant_scale_mode");
`endif

endmodule
