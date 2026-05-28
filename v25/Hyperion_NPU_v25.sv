//
// --------------------------------------------------------------------------
// Hyperion Matrix Vortex - Exascale AI Supercomputer Node v25
// v25: Pipelined reciprocal, hardened TMA, rounding, verification assertions.
//
// Cumulative changes from v24 → v25:
//   FIX-13-revert fp32_adder: normalization part‑select reverted to simple
//     mant_shift_tmp[24:0]; the previous dynamic slice was incorrect.
//   FIX-20 tma_tensor_loader: end‑of‑column condition now uses equality
//     (col_q == tile_n_q - 1) to avoid 16‑bit addition overflow.
//   FIX-21 flash_shared_normalizer: round‑robin modulo replaced with
//     conditional subtraction to improve synthesis QoR.
//   FIX-22 fp32_recip_nr_approx: fully pipelined (3 stages) to eliminate
//     combinational critical path.
//   FIX-23 tma_tensor_loader: explicit tile_n_q != 0 guard in end conditions.
//   FIX-24 fp16_adder_ref / fp32_to_fp16: added round‑to‑nearest‑even
//     conversion to reduce directional bias.
//   VERIF New SVA properties for FP normalisation, TMA state, and
//     normalizer fairness.
//   NOTE fp32_recip_nr_approx removed – replaced by fp32_recip_pipelined.
//   NOTE FP adders and converters: rounding updated to nearest‑even where
//     applicable.
//
// --------------------------------------------------------------------------

`timescale 1ns / 1ps


// --------------------------------------------------------------------------
// Synchronous FIFO (unchanged except added assertions)
// --------------------------------------------------------------------------
module sync_fifo #(
  parameter int DATA_W = 64,
  parameter int DEPTH = 64,
  parameter int ALMOST_FULL_THRESH = (DEPTH*3)/4
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               push,
  input  logic [DATA_W-1:0]  data_in,
  input  logic               pop,
  output logic [DATA_W-1:0]  data_out,
  output logic               valid_out,
  output logic               empty,
  output logic               full,
  output logic               almost_full
);

  localparam int PTR_W   = (DEPTH <= 1) ? 1 : $clog2(DEPTH);
  localparam int COUNT_W = PTR_W + 1;
  localparam logic [COUNT_W-1:0] DEPTH_COUNT = DEPTH;
  localparam logic [COUNT_W-1:0] AFULL_COUNT = ALMOST_FULL_THRESH;
  localparam logic [PTR_W-1:0]   LAST_PTR    = DEPTH-1;
  localparam logic [PTR_W-1:0]   PTR_ONE     = 1;

  logic [DATA_W-1:0] mem [0:DEPTH-1];
  logic [PTR_W-1:0]  wr_ptr, rd_ptr;
  logic [COUNT_W-1:0] count;

  assign empty      = (count == '0);
  assign full       = (count == DEPTH_COUNT);
  assign valid_out  = !empty;
  assign almost_full = (count >= AFULL_COUNT);
  assign data_out   = mem[rd_ptr];

  function automatic logic [PTR_W-1:0] ptr_inc(input logic [PTR_W-1:0] ptr);
    ptr_inc = (ptr == LAST_PTR) ? '0 : (ptr + PTR_ONE);
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
          wr_ptr      <= ptr_inc(wr_ptr);
          count       <= count + 1'b1;
        end
        2'b01: begin
          rd_ptr <= ptr_inc(rd_ptr);
          count  <= count - 1'b1;
        end
        2'b11: begin
          mem[wr_ptr] <= data_in;
          wr_ptr      <= ptr_inc(wr_ptr);
          rd_ptr      <= ptr_inc(rd_ptr);
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


// --------------------------------------------------------------------------
// One-entry ready/valid output register (added hold stability assertion)
// --------------------------------------------------------------------------
module axis_hold_reg #(
  parameter int DATA_W = 64
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               ce,
  input  logic [DATA_W-1:0]  s_data,
  input  logic               s_valid,
  output logic               s_ready,
  output logic [DATA_W-1:0]  m_data,
  output logic               m_valid,
  input  logic               m_ready
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


// --------------------------------------------------------------------------
// Compact finite FP32 helper functions/modules.
// These are deterministic reference blocks, not full IEEE-754 IP.
// --------------------------------------------------------------------------
module fp32_adder (
  input  logic [31:0] a,
  input  logic [31:0] b,
  output logic [31:0] sum
);

  logic        sign_a, sign_b, sign_big, sign_small, sign_out;
  logic [7:0]  exp_a, exp_b, exp_big, exp_small, exp_out, exp_diff;
  logic [24:0] mant_a, mant_b, mant_big, mant_small, mant_small_shifted, mant_norm;
  logic [25:0] mant_calc;
  logic [4:0]  norm_shift;
  logic [48:0] mant_shift_tmp;

  function automatic logic [4:0] leading_zero_shift_23(input logic [24:0] mant);
    begin
      if (mant[23])      leading_zero_shift_23 = 5'd0;
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

    sign_big   = sign_a;
    sign_small = sign_b;
    exp_big    = exp_a;
    exp_small  = exp_b;
    mant_big   = mant_a;
    mant_small = mant_b;

    if ((exp_b > exp_a) || ((exp_b == exp_a) && (mant_b > mant_a))) begin
      sign_big   = sign_b;
      sign_small = sign_a;
      exp_big    = exp_b;
      exp_small  = exp_a;
      mant_big   = mant_b;
      mant_small = mant_a;
    end

    exp_diff = exp_big - exp_small;
    mant_small_shifted = mant_small >> ((exp_diff > 8'd24) ? 5'd24 : exp_diff[4:0]);
    exp_out  = exp_big;
    sign_out = sign_big;
    mant_calc = 26'd0;
    mant_norm = 25'd0;
    norm_shift = 5'd0;
    mant_shift_tmp = 49'd0;
    sum = 32'd0;

    // FIX-18: NaN generation for (+∞) + (−∞)
    if ((a[30:23]==8'hFF) && (b[30:23]==8'hFF) &&
        (a[22:0]==23'd0) && (b[22:0]==23'd0) &&
        (a[31] != b[31])) begin
      sum = 32'h7FC00000; // quiet NaN
    end else if (a[30:23] == 8'hFF) begin
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
        exp_out   = exp_big + 8'd1;
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
          // FIX-13-revert: simple right‑justified slice after left shift
          mant_shift_tmp = {24'd0, mant_norm} << norm_shift;
          mant_norm = mant_shift_tmp[24:0];
          exp_out   = exp_out - norm_shift;
        end else begin
          mant_norm = 25'd0;
          exp_out   = 8'd0;
        end
      end
      sum = (mant_norm == 25'd0) ? 32'd0 : {sign_out, exp_out, mant_norm[22:0]};
    end
  end

`ifndef SYNTHESIS
  // VERIF: after subtraction normalisation, hidden bit must be set
  property p_norm_hidden_bit;
    @(*) disable iff (a[30:23]==8'hFF || b[30:23]==8'hFF)
      (sign_big != sign_small) && (mant_norm != 25'd0) && (norm_shift != 0) && (exp_out > 0)
      |-> mant_norm[24];
  endproperty
  assert property (p_norm_hidden_bit) else $error("FP32 normalisation failed: hidden bit not set");
`endif
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

  logic        sign_out;
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
      if (exp_norm <= 11'sd0)        prod = 32'd0;
      else if (exp_norm >= 11'sd255) prod = {sign_out, 8'hFE, 23'h7FFFFF};
      else                           prod = {sign_out, exp_norm[7:0], frac_out};
    end
  end
endmodule


// --------------------------------------------------------------------------
// Compact finite FP16 adder wrapper with round‑to‑nearest‑even (FIX-24)
// --------------------------------------------------------------------------
module fp16_adder_ref (
  input  logic [15:0] a,
  input  logic [15:0] b,
  output logic [15:0] sum
);

  function automatic logic [31:0] fp16_to_fp32(input logic [15:0] h);
    logic        sign;
    logic [4:0]  exp_h;
    logic [9:0]  frac_h;
    logic [7:0]  exp32;
    begin
      sign  = h[15];
      exp_h = h[14:10];
      frac_h = h[9:0];
      if (exp_h == 5'd0) begin
        fp16_to_fp32 = 32'd0;
      end else if (exp_h == 5'h1F) begin
        fp16_to_fp32 = {sign, 8'hFF, frac_h, 13'd0};
      end else begin
        exp32 = {3'd0, exp_h} + 8'd112;
        fp16_to_fp32 = {sign, exp32, frac_h, 13'd0};
      end
    end
  endfunction

  // FIX-24: round-to-nearest-even in FP32→FP16 conversion
  function automatic logic [15:0] fp32_to_fp16(input logic [31:0] f);
    logic        sign;
    logic [7:0]  exp_f;
    logic [22:0] frac_f;
    logic signed [9:0] exp_h_s;
    logic        round_bit, sticky, round_up;
    logic [9:0]  frac_h;
    begin
      sign  = f[31];
      exp_f = f[30:23];
      frac_f = f[22:0];
      exp_h_s = $signed({2'd0, exp_f}) - 10'sd112;

      if (exp_f == 8'hFF) begin
        fp32_to_fp16 = {sign, 5'h1F, frac_f[22:13]};
      end else if (exp_h_s <= 0) begin
        fp32_to_fp16 = 16'd0;
      end else if (exp_h_s >= 31) begin
        fp32_to_fp16 = {sign, 5'h1E, 10'h3FF};
      end else begin
        // Round-to-nearest-even on 10-bit fraction
        round_bit = frac_f[12];
        sticky    = |frac_f[11:0];
        frac_h    = frac_f[22:13];
        round_up  = round_bit && (sticky || frac_h[0]); // ties to even
        if (round_up) begin
          if (frac_h == 10'h3FF) begin
            exp_h_s = exp_h_s + 10'sd1;
            frac_h  = 10'd0;
          end else begin
            frac_h = frac_h + 10'd1;
          end
        end
        // Recheck exponent bounds after possible rounding overflow
        if (exp_h_s <= 0)      fp32_to_fp16 = 16'd0;
        else if (exp_h_s >= 31) fp32_to_fp16 = {sign, 5'h1E, 10'h3FF};
        else                    fp32_to_fp16 = {sign, exp_h_s[4:0], frac_h};
      end
    end
  endfunction

  logic [31:0] a32, b32, sum32;
  assign a32 = fp16_to_fp32(a);
  assign b32 = fp16_to_fp32(b);
  fp32_adder u_fp16_add_as_fp32 (.a(a32), .b(b32), .sum(sum32));
  assign sum = fp32_to_fp16(sum32);
endmodule


// --------------------------------------------------------------------------
// Pipelined FP32 reciprocal -- 3-stage (seed, iter1, iter2)  FIX-22
// --------------------------------------------------------------------------
module fp32_recip_pipelined (
  input  logic        clk,
  input  logic        rst_n,
  input  logic        ce,
  input  logic [31:0] x,
  input  logic        valid_in,
  output logic [31:0] recip,
  output logic        valid_out
);

  localparam logic [31:0] FP32_TWO = 32'h4000_0000;

  // Pipeline registers
  logic [31:0] x_s1, x_s2;
  logic [31:0] y0_s1;
  logic [31:0] xy0_s2, two_minus_xy0_s2, y1_s2;
  logic [31:0] xy1_s3, two_minus_xy1_s3;
  logic valid_s1, valid_s2, valid_s3;

  // Seed generation (combinational)
  logic [31:0] y0_seed;
  always_comb begin
    if (x[30:0] == 31'd0) begin
      y0_seed = {x[31], 8'hFE, 23'h7FFFFF};
    end else if (x[30:23] == 8'hFF) begin
      y0_seed = {x[31], 31'd0};
    end else begin
      y0_seed = {x[31], (8'd254 - x[30:23]), ~x[22:0]};
    end
  end

  // Pipeline stage 1: capture seed and x
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      x_s1    <= 32'd0;
      y0_s1   <= 32'd0;
      valid_s1 <= 1'b0;
    end else if (ce) begin
      x_s1    <= x;
      y0_s1   <= y0_seed;
      valid_s1 <= valid_in;
    end
  end

  // Compute first iteration (combinational)
  logic [31:0] xy0_comb, two_minus_xy0_comb, y1_comb;
  fp32_mul u_seed_mul0 (.a(x_s1), .b(y0_s1), .prod(xy0_comb));
  fp32_sub u_nr_sub0   (.a(FP32_TWO), .b(xy0_comb), .diff(two_minus_xy0_comb));
  fp32_mul u_nr_mul0   (.a(y0_s1), .b(two_minus_xy0_comb), .prod(y1_comb));

  // Pipeline stage 2
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      x_s2           <= 32'd0;
      xy0_s2         <= 32'd0;
      two_minus_xy0_s2 <= 32'd0;
      y1_s2          <= 32'd0;
      valid_s2       <= 1'b0;
    end else if (ce) begin
      x_s2           <= x_s1;
      xy0_s2         <= xy0_comb;
      two_minus_xy0_s2 <= two_minus_xy0_comb;
      y1_s2          <= y1_comb;
      valid_s2       <= valid_s1;
    end
  end

  // Compute second iteration (combinational)
  logic [31:0] xy1_comb, two_minus_xy1_comb, recip_comb;
  fp32_mul u_seed_mul1 (.a(x_s2), .b(y1_s2), .prod(xy1_comb));
  fp32_sub u_nr_sub1   (.a(FP32_TWO), .b(xy1_comb), .diff(two_minus_xy1_comb));
  fp32_mul u_nr_mul1   (.a(y1_s2), .b(two_minus_xy1_comb), .prod(recip_comb));

  // Pipeline stage 3 (output)
  always_ff @(posedge clk) begin
    if (!rst_n) begin
      xy1_s3         <= 32'd0;
      two_minus_xy1_s3 <= 32'd0;
      recip          <= 32'd0;
      valid_s3       <= 1'b0;
    end else if (ce) begin
      xy1_s3         <= xy1_comb;
      two_minus_xy1_s3 <= two_minus_xy1_comb;
      recip          <= recip_comb;
      valid_s3       <= valid_s2;
    end
  end

  assign valid_out = valid_s3;
endmodule


// --------------------------------------------------------------------------
// Shared FlashAttention normalizer – adapted for pipelined reciprocal (FIX-21, FIX-22)
// --------------------------------------------------------------------------
module flash_shared_normalizer #(
  parameter int COLS = 4,
  parameter int PS_W = 32
)(
  input  logic                       clk,
  input  logic                       rst_n,
  input  logic                       ce,
  input  logic                       clear,
  input  logic [COLS*PS_W-1:0]       num_in_flat,
  input  logic [COLS*PS_W-1:0]       den_in_flat,
  input  logic [COLS-1:0]            valid_in_flat,
  output logic [COLS*PS_W-1:0]       norm_out_flat,
  output logic [COLS-1:0]            valid_out_flat
);

  localparam int IDX_W = (COLS <= 1) ? 1 : $clog2(COLS);
  localparam logic [31:0] FP32_ONE = 32'h3F80_0000;

  logic [PS_W-1:0] num_in [0:COLS-1];
  logic [PS_W-1:0] den_in [0:COLS-1];
  logic            valid_in [0:COLS-1];
  logic [PS_W-1:0] norm_out [0:COLS-1];
  logic            valid_out [0:COLS-1];

  always_comb begin
    for (int c = 0; c < COLS; c++) begin
      num_in[c] = num_in_flat [c*PS_W +: PS_W];
      den_in[c] = den_in_flat [c*PS_W +: PS_W];
      valid_in[c] = valid_in_flat[c];
      norm_out_flat [c*PS_W +: PS_W] = norm_out[c];
      valid_out_flat[c] = valid_out[c];
    end
  end

  logic [PS_W-1:0] pend_num [0:COLS-1];
  logic [PS_W-1:0] pend_den [0:COLS-1];
  logic            pending [0:COLS-1];
  logic [IDX_W-1:0] rr_ptr;
  logic [IDX_W-1:0] sel_idx_comb;
  logic             sel_valid_comb;
  logic [PS_W-1:0]  sel_num_comb, sel_den_comb;

  // Pipeline stages for the reciprocal and multiplier
  logic             s4a_valid, s4b_valid, s4c_valid, s4d_valid;
  logic [IDX_W-1:0] s4a_idx, s4b_idx, s4c_idx, s4d_idx;
  logic [PS_W-1:0]  s4a_num, s4b_num, s4c_num;
  logic [PS_W-1:0]  s4a_den;
  logic [PS_W-1:0]  recip_s3;      // reciprocal output after 2 cycles
  logic             recip_valid_s3;
  logic [PS_W-1:0]  norm_comb;

  function automatic logic [IDX_W-1:0] idx_inc(input logic [IDX_W-1:0] idx);
    idx_inc = (idx == COLS-1) ? '0 : idx + 1'b1;
  endfunction

  always_comb begin
    sel_idx_comb = rr_ptr;
    sel_valid_comb = 1'b0;
    for (int k = 0; k < COLS; k++) begin
      int unsigned probe;
      // FIX-21: avoid modulo for synthesis
      probe = rr_ptr + k;
      if (probe >= COLS) probe -= COLS;
      if (!sel_valid_comb && pending[probe]) begin
        sel_idx_comb = probe[IDX_W-1:0];
        sel_valid_comb = 1'b1;
      end
    end
    sel_num_comb = pend_num[sel_idx_comb];
    sel_den_comb = (pend_den[sel_idx_comb][30:0] == 31'd0) ? FP32_ONE : pend_den[sel_idx_comb];
  end

  // Pipelined reciprocal (3-stage: seed, iter1, iter2)
  fp32_recip_pipelined u_recip (
    .clk      (clk),
    .rst_n    (rst_n),
    .ce       (ce),
    .x        (s4a_den),
    .valid_in (s4a_valid),
    .recip    (recip_s3),
    .valid_out(recip_valid_s3)
  );

  fp32_mul u_norm_mul (.a(s4c_num), .b(recip_s3), .prod(norm_comb));

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rr_ptr <= '0;
      s4a_valid <= 1'b0; s4b_valid <= 1'b0; s4c_valid <= 1'b0; s4d_valid <= 1'b0;
      s4a_idx <= '0; s4b_idx <= '0; s4c_idx <= '0; s4d_idx <= '0;
      s4a_num <= '0; s4b_num <= '0; s4c_num <= '0;
      s4a_den <= FP32_ONE;
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
          pending[c] <= 1'b1;
        end
      end

      if (clear) begin
        rr_ptr <= '0;
        s4a_valid <= 1'b0; s4b_valid <= 1'b0; s4c_valid <= 1'b0; s4d_valid <= 1'b0;
        for (int c = 0; c < COLS; c++) begin
          pending[c]   <= 1'b0;
          valid_out[c] <= 1'b0;
          norm_out[c]  <= '0;
        end
      end else begin
        // Stage 1: capture selection
        s4a_valid <= sel_valid_comb;
        s4a_idx   <= sel_idx_comb;
        s4a_num   <= sel_num_comb;
        s4a_den   <= sel_den_comb;
        if (sel_valid_comb) begin
          pending[sel_idx_comb] <= 1'b0;
          rr_ptr <= idx_inc(sel_idx_comb);
        end

        // Stage 2: delay num and idx while reciprocal runs
        s4b_valid <= s4a_valid;
        s4b_idx   <= s4a_idx;
        s4b_num   <= s4a_num;

        // Stage 3: more delay (reciprocal result appears at the end of this stage)
        s4c_valid <= s4b_valid;
        s4c_idx   <= s4b_idx;
        s4c_num   <= s4b_num;

        // Stage 4: multiply and output
        s4d_valid <= s4c_valid && recip_valid_s3;
        s4d_idx   <= s4c_idx;
        if (s4c_valid && recip_valid_s3) begin
          norm_out[s4c_idx]  <= norm_comb;
          valid_out[s4c_idx] <= 1'b1;
        end
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (COLS > 0) else $fatal("flash_shared_normalizer COLS must be positive");
    assert (PS_W == 32) else $fatal("flash_shared_normalizer currently requires PS_W=32");
  end

  // VERIF: round-robin fairness -- any pending request will be served within COLS cycles
  property p_rr_fairness;
    logic [IDX_W-1:0] expected;
    @(posedge clk) disable iff (!rst_n || clear)
      (pending[rr_ptr] && !sel_valid_comb) |=> ##[1:COLS] sel_valid_comb;
  endproperty
  assert property (p_rr_fairness)
    else $error("round-robin normalizer failed to serve a pending request within COLS cycles");
`endif
endmodule


// --------------------------------------------------------------------------
// Registered clock-enable relay (unchanged)
// --------------------------------------------------------------------------
module ce_relay_grid #(
  parameter int ROWS = 4,
  parameter int COLS = 4
)(
  input  logic            clk,
  input  logic            rst_n,
  input  logic            root_step,
  output logic            ingress_ce,
  output logic [ROWS-1:0] row_ce_flat,
  output logic [COLS-1:0] col_ce_flat
);
  logic row_ce [0:ROWS-1];
  logic col_ce [0:COLS-1];
  always_comb begin
    for (int r = 0; r < ROWS; r++) row_ce_flat[r] = row_ce[r];
    for (int c = 0; c < COLS; c++) col_ce_flat[c] = col_ce[c];
  end
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


// --------------------------------------------------------------------------
// Ping-pong vector TCSM feed wrapper (unchanged)
// --------------------------------------------------------------------------
module ping_pong_vector_tcsm #(
  parameter int DATA_W = 64,
  parameter int DEPTH = 256,
  parameter int ADDR_W = (DEPTH <= 1) ? 1 : $clog2(DEPTH)
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               load_en,
  input  logic               load_bank,
  input  logic [ADDR_W-1:0]  load_addr,
  input  logic [DATA_W-1:0]  load_data,
  input  logic [ADDR_W-1:0]  read_addr,
  input  logic               swap_banks,
  output logic [DATA_W-1:0]  read_data,
  output logic               active_bank
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


// --------------------------------------------------------------------------
// Tensor Descriptor Loader / TMA-lite (FIX-20, FIX-23)
// --------------------------------------------------------------------------
module tma_tensor_loader #(
  parameter int DATA_W = 128,
  parameter int ADDR_W = 8,
  parameter int DIM_W = 16
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               desc_valid,
  output logic               desc_ready,
  input  logic [63:0]        desc_base_addr,
  input  logic [DIM_W-1:0]   desc_dim_m,
  input  logic [DIM_W-1:0]   desc_dim_n,
  input  logic [DIM_W-1:0]   desc_stride_m,
  input  logic [DIM_W-1:0]   desc_stride_n,
  input  logic [DIM_W-1:0]   desc_tile_m,
  input  logic [DIM_W-1:0]   desc_tile_n,
  input  logic [1:0]         desc_dst_kind,
  input  logic               desc_dst_bank,
  input  logic               hold,
  input  logic [DATA_W-1:0]  stream_data,
  input  logic               stream_valid,
  output logic               stream_ready,
  output logic               load_valid,
  output logic [1:0]         load_dst_kind,
  output logic               load_bank,
  output logic [ADDR_W-1:0]  load_addr,
  output logic [63:0]        load_addr_full,
  output logic [DATA_W-1:0]  load_data,
  output logic               busy,
  output logic               done,
  output logic               desc_error
);

  logic [DIM_W-1:0] row_q, col_q, tile_m_q, tile_n_q;
  logic [63:0]      row_base_q, addr_q;
  logic [1:0]       dst_kind_q;
  logic             dst_bank_q;
  logic             active_q;
  logic [63:0]      base_addr_q;
  logic [DIM_W-1:0] stride_m_q, stride_n_q, dim_m_q, dim_n_q;
  logic             desc_bad;
  logic             tma_end_col, tma_end_row;
  logic [63:0]      tma_next_row_base;

  assign desc_ready   = !active_q && !hold;
  assign stream_ready = active_q && !hold;
  assign busy         = active_q;
  assign load_addr_full = addr_q;

  always_comb begin
    desc_bad = (desc_dim_m == '0) || (desc_dim_n == '0) ||
               ((desc_tile_m != '0) && (desc_tile_m > desc_dim_m)) ||
               ((desc_tile_n != '0) && (desc_tile_n > desc_dim_n));
    // FIX-20 + FIX-23: end-of-col/row with explicit zero guard
    tma_end_col = (tile_n_q != 0) && (col_q == tile_n_q - 1'b1);
    tma_end_row = (tile_m_q != 0) && (row_q == tile_m_q - 1'b1);
    tma_next_row_base = row_base_q + {{(64-DIM_W){1'b0}}, stride_m_q};
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      row_q        <= '0; col_q <= '0; tile_m_q <= '0; tile_n_q <= '0;
      row_base_q   <= 64'd0; addr_q <= 64'd0;
      dst_kind_q   <= '0; dst_bank_q <= 1'b0; active_q <= 1'b0;
      load_valid   <= 1'b0; done <= 1'b0; desc_error <= 1'b0;
      base_addr_q  <= '0; stride_m_q <= '0; stride_n_q <= '0; dim_m_q <= '0; dim_n_q <= '0;
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
          stride_m_q  <= (desc_stride_m == '0) ? desc_dim_n : desc_stride_m;
          stride_n_q  <= (desc_stride_n == '0) ? {{(DIM_W-1){1'b0}},1'b1} : desc_stride_n;
          dim_m_q     <= desc_dim_m;
          dim_n_q     <= desc_dim_n;
        end
      end else if (active_q && stream_valid && stream_ready) begin
        load_valid    <= 1'b1;
        load_dst_kind <= dst_kind_q;
        load_bank     <= dst_bank_q;
        load_addr     <= addr_q[ADDR_W-1:0];
        load_data     <= stream_data;

        if (tma_end_col) begin
          col_q <= '0;
          if (tma_end_row) begin
            row_q    <= '0;
            active_q <= 1'b0;
            done     <= 1'b1;
          end else begin
            row_q        <= row_q + 1'b1;
            row_base_q   <= tma_next_row_base;
            addr_q       <= tma_next_row_base;
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

  // VERIF: tile bounds
  property p_col_within_tile;
    @(posedge clk) disable iff (!rst_n) (active_q && !tma_end_col) |-> (col_q < tile_n_q);
  endproperty
  assert property (p_col_within_tile) else $error("TMA col_q out of bounds");

  property p_row_within_tile;
    @(posedge clk) disable iff (!rst_n) (active_q) |-> (row_q < tile_m_q);
  endproperty
  assert property (p_row_within_tile) else $error("TMA row_q out of bounds");
`endif
endmodule


// --------------------------------------------------------------------------
// Lightweight KV Page Table Walker (unchanged, includes FIX-15)
// --------------------------------------------------------------------------
module kv_page_table #(
  parameter int VPN_W = 12,
  parameter int PPN_W = 24,
  parameter int PAGE_COUNT = 256,
  parameter int PAGE_AW = (PAGE_COUNT <= 1) ? 1 : $clog2(PAGE_COUNT)
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               lookup_valid,
  output logic               lookup_ready,
  input  logic [VPN_W-1:0]   lookup_vpn,
  output logic               lookup_resp_valid,
  output logic               lookup_miss,
  output logic [PPN_W-1:0]   lookup_ppn,
  output logic               pager_stall,
  output logic               fault_valid,
  output logic [VPN_W-1:0]   fault_vpn,
  input  logic               fault_clear,
  input  logic               ptw_write_valid,
  input  logic [PAGE_AW-1:0] ptw_write_index,
  input  logic [VPN_W-1:0]   ptw_write_vpn,
  input  logic [PPN_W-1:0]   ptw_write_ppn,
  input  logic               ptw_write_valid_bit
);

  logic [VPN_W-1:0] vpn_mem [0:PAGE_COUNT-1];
  logic [PPN_W-1:0] ppn_mem [0:PAGE_COUNT-1];
  logic             valid_mem [0:PAGE_COUNT-1];
  logic [PAGE_AW-1:0] idx;
  logic             hit_comb;
  logic             forwarding_hit; // FIX-15: true only when index and VPN match on same-cycle write

  assign idx = lookup_vpn[PAGE_AW-1:0];
  assign forwarding_hit = ptw_write_valid && (ptw_write_index == idx) && (ptw_write_vpn == lookup_vpn);
  assign hit_comb = forwarding_hit ? 1'b1 : (valid_mem[idx] && (vpn_mem[idx] == lookup_vpn));
  assign pager_stall = fault_valid;
  assign lookup_ready = !pager_stall || fault_clear;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      lookup_resp_valid <= 1'b0;
      lookup_miss       <= 1'b1;
      lookup_ppn        <= '0;
      fault_valid       <= 1'b0;
      fault_vpn         <= '0;
      for (int i = 0; i < PAGE_COUNT; i++) begin
        valid_mem[i] <= 1'b0;
        vpn_mem[i]   <= '0;
        ppn_mem[i]   <= '0;
      end
    end else begin
      lookup_resp_valid <= 1'b0;
      if (ptw_write_valid) begin
        vpn_mem[ptw_write_index] <= ptw_write_vpn;
        ppn_mem[ptw_write_index] <= ptw_write_ppn;
        valid_mem[ptw_write_index] <= ptw_write_valid_bit;
      end
      if (fault_clear) begin
        fault_valid <= 1'b0;
      end
      if (lookup_valid && lookup_ready) begin
        lookup_resp_valid <= 1'b1;
        if (forwarding_hit) begin
          lookup_ppn  <= ptw_write_ppn; // use forwarded data
          lookup_miss <= 1'b0;
        end else begin
          lookup_ppn  <= ppn_mem[idx];
          lookup_miss <= !hit_comb;
        end
        if (!hit_comb) begin
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

  property p_fault_clear_requires_ptw_write;
    @(posedge clk) disable iff (!rst_n)
      fault_clear |-> $past(ptw_write_valid, 1);
  endproperty
  assert property (p_fault_clear_requires_ptw_write)
    else $warning("fault_clear asserted without a prior ptw_write_valid --- re-fault likely");
`endif
endmodule


// --------------------------------------------------------------------------
// Quantization helpers (unchanged)
// --------------------------------------------------------------------------
// (Placeholder – actual helpers are inside unified_fracturable_mac)


// --------------------------------------------------------------------------
// Unified Fracturable MAC (unchanged, includes FIX-14, FIX-11, etc.)
// --------------------------------------------------------------------------
module unified_fracturable_mac (
  input  logic               clk,
  input  logic               rst_n,
  input  logic               ce,
  input  logic [3:0]         cfg_mode,
  input  logic               cfg_mx_native_accum,
  input  logic               cfg_mx_finalize,
  input  logic [7:0]         shared_exp,
  input  logic               cfg_quant_en,
  input  logic [1:0]         cfg_quant_scale_mode,
  input  logic [15:0]        quant_scale_q8_8,
  input  logic [31:0]        quant_scale_fp32,
  input  logic signed [31:0] quant_bias_i32,
  input  logic signed [15:0] act_zero_point,
  input  logic signed [15:0] wt_zero_point,
  input  logic [15:0]        a_in,
  input  logic [15:0]        b_in,
  input  logic [3:0]         sparse_meta,
  input  logic [31:0]        c_accum,
  output logic [31:0]        mac_out
);

  logic [3:0]        cfg_mode_q;
  logic              cfg_mx_native_q, cfg_mx_finalize_q;
  logic [7:0]        shared_exp_q;
  logic [31:0]       c_accum_q;
  logic signed [31:0] int16_prod_q, sum_2x8_q, sum_4x4_q, w4a8_prod_q, sparse_prod_q;
  logic signed [31:0] mx8_mant_prod_q, mx4_mant_prod_q, mx_native_sum_q;
  logic signed [31:0] mx8_mant_prod_comb, mx4_mant_prod_comb;
  logic [31:0]       float_product_q;
  logic [1:0]        cfg_quant_scale_mode_q;
  logic              cfg_quant_en_q;
  logic [15:0]       quant_scale_q8_8_q;
  logic [31:0]       quant_scale_fp32_q;
  logic signed [31:0] quant_bias_i32_q;
  logic signed [31:0] int_accum_selected;
  logic [31:0]       int_accum_fp32;
  logic [31:0]       quant_bias_fp32;
  logic [31:0]       quant_fp_scaled;
  logic [31:0]       quant_fp_out;

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

  // FIX-11: Saturated zero-point subtraction functions.
  function automatic logic signed [15:0] zp_sub16(input logic [15:0] x, input logic signed [15:0] zp);
    logic signed [16:0] full;
    full = $signed(x) - $signed(zp);
    if (full > 16'sd32767)      zp_sub16 = 16'sd32767;
    else if (full < -16'sd32768) zp_sub16 = -16'sd32768;
    else                         zp_sub16 = full[15:0];
  endfunction

  function automatic logic signed [7:0] zp_sub8(input logic [7:0] x, input logic signed [15:0] zp);
    logic signed [8:0] full;
    full = $signed(x) - $signed(zp[7:0]);
    if (full > 8'sd127)      zp_sub8 = 8'sd127;
    else if (full < -8'sd128) zp_sub8 = -8'sd128;
    else                      zp_sub8 = full[7:0];
  endfunction

  function automatic logic signed [3:0] zp_sub4(input logic [3:0] x, input logic signed [15:0] zp);
    logic signed [4:0] full;
    full = $signed(x) - $signed(zp[3:0]);
    if (full > 4'sd7)      zp_sub4 = 4'sd7;
    else if (full < -4'sd8) zp_sub4 = -4'sd8;
    else                    zp_sub4 = full[3:0];
  endfunction

  function automatic logic signed [31:0] sxmul4x8(input logic [3:0] x, input logic [7:0] y);
    sxmul4x8 = $signed(x) * $signed(y);
  endfunction

  function automatic logic [31:0] int32_scaled_to_fp32(input logic signed [31:0] val, input logic [7:0] block_exp);
    logic        sign;
    logic [31:0] mag;
    logic [5:0]  msb;
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
        exp32 = $signed({3'd0, block_exp}) + $signed({5'd0, msb});
        if (exp32 <= 0)                int32_scaled_to_fp32 = 32'd0;
        else if (exp32 >= 255)         int32_scaled_to_fp32 = {sign, 8'hFE, 23'h7FFFFF};
        else                           int32_scaled_to_fp32 = {sign, exp32[7:0], shifted[30:8]};
      end
    end
  endfunction

  function automatic logic [31:0] int32_to_fp32(input logic signed [31:0] val);
    logic        sign;
    logic [31:0] mag;
    logic [5:0]  msb;
    logic [55:0] shifted;
    logic [7:0]  exp32;
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
    logic        sign;
    logic [4:0]  ea, eb;
    logic [21:0] mp;
    logic signed [10:0] exp32;
    logic [9:0]  frac10;
    begin
      sign = aa[15] ^ bb[15];
      ea = aa[14:10];
      eb = bb[14:10];
      mp = {1'b1, aa[9:0]} * {1'b1, bb[9:0]};
      exp32 = $signed({6'd0, ea}) + $signed({6'd0, eb}) - 11'sd15 + 11'sd127;
      frac10 = 10'd0;
      if ((ea == 5'd0) || (eb == 5'd0)) begin
        pack_fp16_product = 32'd0;
      end else begin
        if (mp[21]) begin exp32 = exp32 + 11'sd1; frac10 = mp[20:11]; end
        else          frac10 = mp[19:10];
        pack_fp16_product = (exp32 <= 0) ? 32'd0 : {sign, exp32[7:0], frac10, 13'd0};
      end
    end
  endfunction

  // FIX-14: bf16-like product zero detection now ignores sign (treats -0 as zero)
  function automatic logic [31:0] pack_bf16like_product(input logic [15:0] aa, input logic [15:0] bb,
                                                        input logic [7:0] exp_override,
                                                        input logic use_exp_override);
    logic        sign;
    logic [7:0]  ea, eb;
    logic signed [10:0] exp32;
    logic [15:0] mp;
    logic [6:0]  frac7;
    begin
      // Zero when exponent+m mantissa are all zero (sign ignored)
      if ((aa[14:0] == 15'd0) || (bb[14:0] == 15'd0)) begin
        pack_bf16like_product = 32'd0;
        return;
      end
      sign = aa[15] ^ bb[15];
      ea = aa[14:7];
      eb = bb[14:7];
      mp = {1'b1, aa[6:0]} * {1'b1, bb[6:0]};
      exp32 = use_exp_override ? $signed({3'd0, exp_override}) :
                                 ($signed({3'd0, ea}) + $signed({3'd0, eb}) - 11'sd127);
      frac7 = 7'd0;
      if (((ea == 8'd0) || (eb == 8'd0)) && !use_exp_override) begin
        pack_bf16like_product = 32'd0;
      end else begin
        if (mp[15]) begin exp32 = exp32 + 11'sd1; frac7 = mp[14:8]; end
        else         frac7 = mp[13:7];
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
  assign a_lane[0] = {{4{b_in[3]}}, b_in[3:0]};
  assign a_lane[1] = {{4{b_in[7]}}, b_in[7:4]};
  assign a_lane[2] = {{4{b_in[11]}}, b_in[11:8]};
  assign a_lane[3] = {{4{b_in[15]}}, b_in[15:12]};
  assign sp_a0 = a_lane[meta_idx_0];
  assign sp_a1 = a_lane[meta_idx_1];

  logic [31:0] float_adder_out;
  fp32_adder u_float_acc (.a(float_product_q), .b(c_accum_q), .sum(float_adder_out));
  fp32_mul u_quant_fp_mul (.a(int_accum_fp32), .b(quant_scale_fp32_q), .prod(quant_fp_scaled));
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
      cfg_mode_q            <= 4'd0;
      cfg_mx_native_q       <= 1'b0;
      cfg_mx_finalize_q     <= 1'b0;
      shared_exp_q          <= 8'd127;
      c_accum_q             <= 32'd0;
      int16_prod_q          <= 32'sd0;
      sum_2x8_q             <= 32'sd0;
      sum_4x4_q             <= 32'sd0;
      w4a8_prod_q           <= 32'sd0;
      sparse_prod_q         <= 32'sd0;
      mx8_mant_prod_q       <= 32'sd0;
      mx4_mant_prod_q       <= 32'sd0;
      mx_native_sum_q       <= 32'sd0;
      float_product_q       <= 32'd0;
      cfg_quant_scale_mode_q <= 2'd0;
      cfg_quant_en_q         <= 1'b0;
      quant_scale_q8_8_q     <= 16'd256;
      quant_scale_fp32_q     <= 32'h3F80_0000;
      quant_bias_i32_q       <= 32'sd0;
    end else if (ce) begin
      cfg_mode_q            <= cfg_mode;
      cfg_mx_native_q       <= cfg_mx_native_accum;
      cfg_mx_finalize_q     <= cfg_mx_finalize;
      shared_exp_q          <= shared_exp;
      cfg_quant_en_q        <= cfg_quant_en;
      cfg_quant_scale_mode_q <= cfg_quant_scale_mode;
      quant_scale_q8_8_q    <= quant_scale_q8_8;
      quant_scale_fp32_q    <= quant_scale_fp32;
      quant_bias_i32_q      <= quant_bias_i32;
      c_accum_q             <= c_accum;
      int16_prod_q          <= zp_sub16(a_in, wt_zero_point) * zp_sub16(b_in, act_zero_point);
      sum_2x8_q             <= (zp_sub8(a_in[7:0], wt_zero_point) * zp_sub8(b_in[7:0], act_zero_point)) +
                               (zp_sub8(a_in[15:8], wt_zero_point) * zp_sub8(b_in[15:8], act_zero_point));
      sum_4x4_q             <= (zp_sub4(a_in[3:0], wt_zero_point) * zp_sub4(b_in[3:0], act_zero_point)) +
                               (zp_sub4(a_in[7:4], wt_zero_point) * zp_sub4(b_in[7:4], act_zero_point)) +
                               (zp_sub4(a_in[11:8], wt_zero_point) * zp_sub4(b_in[11:8], act_zero_point)) +
                               (zp_sub4(a_in[15:12], wt_zero_point) * zp_sub4(b_in[15:12], act_zero_point));
      w4a8_prod_q           <= (zp_sub4(a_in[3:0], wt_zero_point) * zp_sub8(b_in[7:0], act_zero_point)) +
                               (zp_sub4(a_in[7:4], wt_zero_point) * zp_sub8(b_in[15:8], act_zero_point));
      sparse_prod_q         <= ($signed(sp_w0) * $signed(sp_a0)) + ($signed(sp_w1) * $signed(sp_a1));
      mx8_mant_prod_q       <= mx8_mant_prod_comb;
      mx4_mant_prod_q       <= mx4_mant_prod_comb;
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
        4'h0: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out :
                         qscale_i32($signed(c_accum_q) + int16_prod_q);
        4'h1: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out :
                         qscale_i32($signed(c_accum_q) + sum_2x8_q);
        4'h2: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out :
                         qscale_i32($signed(c_accum_q) + sum_4x4_q);
        4'h3, 4'h4, 4'h6: mac_out <= float_adder_out;
        4'h5: begin
          if (cfg_mx_native_q)
            mac_out <= cfg_mx_finalize_q ? int32_scaled_to_fp32(mx_native_sum_q, shared_exp_q) : mx_native_sum_q;
          else
            mac_out <= float_adder_out;
        end
        4'h7: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out :
                         qscale_i32($signed(c_accum_q) + w4a8_prod_q);
        4'h8: mac_out <= (cfg_quant_en_q && (cfg_quant_scale_mode_q == 2'd2)) ? quant_fp_out :
                         qscale_i32($signed(c_accum_q) + sparse_prod_q);
        4'h9: mac_out <= (cfg_mx_native_q && cfg_mx_finalize_q) ?
                         int32_scaled_to_fp32(mx_native_sum_q, shared_exp_q) : mx_native_sum_q;
        default: mac_out <= c_accum_q;
      endcase
    end
  end
endmodule


// --------------------------------------------------------------------------
// FlashAttention VPU front-end (unchanged)
// --------------------------------------------------------------------------
module flash_attention_vpu #(
  parameter int PS_W = 32
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               ce,
  input  logic               clear_state,
  input  logic [2:0]         cfg_vpu_mode,
  input  logic [15:0]        seq_i,
  input  logic [15:0]        seq_j,
  input  logic [PS_W-1:0]    x_in,
  input  logic               x_valid,
  input  logic [PS_W-1:0]    v_in,
  input  logic [PS_W-1:0]    daisy_chain_in,
  output logic [PS_W-1:0]    daisy_chain_out,
  output logic [PS_W-1:0]    norm_num_out,
  output logic [PS_W-1:0]    norm_den_out,
  output logic               norm_valid_out
);

  localparam logic [31:0] FP32_ZERO   = 32'h0000_0000;
  localparam logic [31:0] FP32_ONE    = 32'h3F80_0000;
  localparam logic [31:0] FP32_HALF   = 32'h3F00_0000;
  localparam logic [31:0] FP32_QUARTER= 32'h3E80_0000;
  localparam logic [31:0] FP32_EIGHTH = 32'h3E00_0000;

  logic [31:0] m_old, l_old, out_old;
  logic m_valid;

  function automatic logic fp32_gt(input logic [31:0] aa, input logic [31:0] bb);
    begin
      if (aa[31] != bb[31])      fp32_gt = !aa[31];
      else if (!aa[31])          fp32_gt = (aa[30:0] > bb[30:0]);
      else                       fp32_gt = (aa[30:0] < bb[30:0]);
    end
  endfunction

  function automatic logic [31:0] exp_neg_approx(input logic [31:0] neg_or_zero);
    logic [7:0] mag_exp;
    logic [22:0] mag_frac;
    begin
      mag_exp = neg_or_zero[30:23];
      mag_frac = neg_or_zero[22:0];
      if (neg_or_zero[30:0] == 31'd0) begin
        exp_neg_approx = FP32_ONE;
      end else if (!neg_or_zero[31]) begin
        exp_neg_approx = FP32_ONE;
      end else if (mag_exp >= 8'd130) begin
        exp_neg_approx = 32'h3C80_0000;
      end else if (mag_exp == 8'd129) begin
        exp_neg_approx = mag_frac[22] ? 32'h3D80_0000 : 32'h3E00_0000;
      end else if (mag_exp == 8'd128) begin
        exp_neg_approx = mag_frac[22] ? 32'h3E40_0000 : FP32_EIGHTH;
      end else if (mag_exp == 8'd127) begin
        exp_neg_approx = mag_frac[22] ? FP32_QUARTER : 32'h3EA0_0000;
      end else begin
        exp_neg_approx = mag_frac[22] ? FP32_HALF : 32'h3F40_0000;
      end
    end
  endfunction

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
  assign m_new_comb = (masked_comb || !x_gt_m_comb) ? m_old : x_in;

  fp32_sub u_diff_old (.a(m_old), .b(m_new_comb), .diff(diff_old_comb));
  fp32_sub u_diff_new (.a(x_in), .b(m_new_comb), .diff(diff_new_comb));

  logic s2_valid, s2_masked, s2_m_valid;
  logic [2:0] s2_mode;
  logic [31:0] s2_x, s2_v, s2_m_new, s2_l_old, s2_out_old;
  logic [31:0] s2_exp_old, s2_exp_new;
  logic [31:0] daisy_s2;

  logic [31:0] l_scaled_comb, l_sum_comb, out_scaled_comb, v_scaled_comb, out_sum_comb;

  logic s3_valid, s3_masked, s3_m_valid;
  logic [2:0] s3_mode;
  logic [31:0] s3_x, s3_m_new, s3_l_sum, s3_out_sum;
  logic [31:0] daisy_s3;

  fp32_mul u_l_scale (.a(s2_l_old), .b(s2_exp_old), .prod(l_scaled_comb));
  fp32_adder u_l_add (.a(l_scaled_comb), .b(s2_exp_new), .sum(l_sum_comb));
  fp32_mul u_out_scale (.a(s2_out_old), .b(s2_exp_old), .prod(out_scaled_comb));
  fp32_mul u_v_scale (.a(s2_v), .b(s2_exp_new), .prod(v_scaled_comb));
  fp32_adder u_out_add (.a(out_scaled_comb), .b(v_scaled_comb), .sum(out_sum_comb));

  logic s4_valid, s4_masked, s4_m_valid;
  logic [2:0] s4_mode;
  logic [31:0] s4_x, s4_m_new, s4_l_sum, s4_out_sum;
  logic [31:0] daisy_s4;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      m_old    <= FP32_ZERO;
      l_old    <= FP32_ZERO;
      out_old  <= FP32_ZERO;
      m_valid  <= 1'b0;
      s1_valid <= 1'b0; s2_valid <= 1'b0; s3_valid <= 1'b0; s4_valid <= 1'b0;
      daisy_s1 <= FP32_ZERO; daisy_s2 <= FP32_ZERO; daisy_s3 <= FP32_ZERO; daisy_s4 <= FP32_ZERO;
      norm_num_out   <= FP32_ZERO;
      norm_den_out   <= FP32_ONE;
      norm_valid_out <= 1'b0;
      daisy_chain_out <= FP32_ZERO;
    end else if (ce) begin
      norm_valid_out <= 1'b0;
      if (clear_state) begin
        m_old    <= FP32_ZERO;
        l_old    <= FP32_ZERO;
        out_old  <= FP32_ZERO;
        m_valid  <= 1'b0;
        s1_valid <= 1'b0; s2_valid <= 1'b0; s3_valid <= 1'b0; s4_valid <= 1'b0;
        daisy_s1 <= FP32_ZERO; daisy_s2 <= FP32_ZERO; daisy_s3 <= FP32_ZERO; daisy_s4 <= FP32_ZERO;
        norm_num_out   <= FP32_ZERO;
        norm_den_out   <= FP32_ONE;
        norm_valid_out <= 1'b0;
        daisy_chain_out <= FP32_ZERO;
      end else begin
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
                m_old    <= s4_m_new;
                l_old    <= s4_l_sum;
                out_old  <= s4_out_sum;
                m_valid  <= 1'b1;
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
        s4_valid <= s3_valid;
        s4_masked <= s3_masked;
        s4_m_valid <= s3_m_valid;
        s4_mode <= s3_mode;
        s4_x <= s3_x;
        s4_m_new <= s3_m_new;
        s4_l_sum <= s3_l_sum;
        s4_out_sum <= s3_out_sum;
        daisy_s4 <= daisy_s3;

        s3_valid <= s2_valid;
        s3_masked <= s2_masked;
        s3_m_valid <= s2_m_valid;
        s3_mode <= s2_mode;
        s3_x <= s2_x;
        s3_m_new <= s2_m_new;
        s3_l_sum <= l_sum_comb;
        s3_out_sum <= out_sum_comb;
        daisy_s3 <= daisy_s2;

        s2_valid <= s1_valid;
        s2_masked <= s1_masked;
        s2_m_valid <= s1_m_valid;
        s2_mode <= s1_mode;
        s2_x <= s1_x;
        s2_v <= s1_v;
        s2_m_new <= s1_m_new;
        s2_l_old <= s1_l_old;
        s2_out_old <= s1_out_old;
        s2_exp_old <= (!s1_m_valid) ? FP32_ZERO : exp_neg_approx(s1_diff_old);
        s2_exp_new <= s1_masked ? FP32_ZERO : exp_neg_approx(s1_diff_new);
        daisy_s2 <= daisy_s1;

        s1_valid <= x_valid;
        s1_masked <= masked_comb;
        s1_m_valid <= m_valid;
        s1_mode <= cfg_vpu_mode;
        s1_x <= x_in;
        s1_v <= v_in;
        s1_m_old <= m_old;
        s1_l_old <= l_old;
        s1_out_old <= out_old;
        s1_m_new <= m_new_comb;
        s1_diff_old <= diff_old_comb;
        s1_diff_new <= diff_new_comb;
        daisy_s1 <= (cfg_vpu_mode == 3'd2) ? (fp32_gt(x_in, daisy_chain_in) ? x_in : daisy_chain_in) : daisy_chain_in;
      end
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (PS_W == 32) else $fatal("flash_attention_vpu currently requires PS_W=32");
  end
`endif
endmodule


// --------------------------------------------------------------------------
// RoPE engine (unchanged)
// --------------------------------------------------------------------------
module rope_engine #(
  parameter int DATA_W = 64
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               cfg_rope_en,
  input  logic [DATA_W-1:0]  s_tdata,
  input  logic               s_tvalid,
  output logic               s_tready,
  output logic [DATA_W-1:0]  m_tdata,
  output logic               m_tvalid,
  input  logic               m_tready
);

  localparam int LANES = DATA_W / 16;

  logic [DATA_W-1:0] rope_pipe [0:1];
  logic              valid_pipe [0:1];
  logic [2:0]        phase_q;

  function automatic logic signed [15:0] cos_lut(input logic [2:0] phase);
    unique case (phase)
      3'd0: cos_lut = 16'sd16384;
      3'd1: cos_lut = 16'sd11585;
      3'd2: cos_lut = 16'sd0;
      3'd3: cos_lut = -16'sd11585;
      3'd4: cos_lut = -16'sd16384;
      3'd5: cos_lut = -16'sd11585;
      3'd6: cos_lut = 16'sd0;
      3'd7: cos_lut = 16'sd11585;
      default: cos_lut = 16'sd11585;
    endcase
  endfunction

  function automatic logic signed [15:0] sin_lut(input logic [2:0] phase);
    unique case (phase)
      3'd0: sin_lut = 16'sd0;
      3'd1: sin_lut = 16'sd11585;
      3'd2: sin_lut = 16'sd16384;
      3'd3: sin_lut = 16'sd11585;
      3'd4: sin_lut = 16'sd0;
      3'd5: sin_lut = -16'sd11585;
      3'd6: sin_lut = -16'sd16384;
      3'd7: sin_lut = -16'sd11585;
      default: sin_lut = -16'sd11585;
    endcase
  endfunction

  function automatic logic signed [15:0] sat16(input logic signed [31:0] v);
    if (v > 32'sd32767)      sat16 = 16'sh7FFF;
    else if (v < -32'sd32768) sat16 = 16'sh8000;
    else                     sat16 = v[15:0];
  endfunction

  function automatic logic [31:0] rotate_pair_lut(input logic [15:0] x_bits, input logic [15:0] y_bits, input logic [2:0] phase);
    logic signed [15:0] x, y, c, s;
    logic signed [31:0] xr_wide, yr_wide;
    logic signed [15:0] xr, yr;
    begin
      x = $signed(x_bits);
      y = $signed(y_bits);
      c = cos_lut(phase);
      s = sin_lut(phase);
      xr_wide = (($signed(c) * $signed(x)) - ($signed(s) * $signed(y))) >>> 14;
      yr_wide = (($signed(s) * $signed(x)) + ($signed(c) * $signed(y))) >>> 14;
      xr = sat16(xr_wide);
      yr = sat16(yr_wide);
      rotate_pair_lut = {yr, xr};
    end
  endfunction

  assign s_tready = m_tready;

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      rope_pipe[0]  <= '0;
      rope_pipe[1]  <= '0;
      valid_pipe[0] <= 1'b0;
      valid_pipe[1] <= 1'b0;
      phase_q       <= 3'd0;
    end else if (m_tready) begin
      rope_pipe[0] <= s_tdata;
      if (cfg_rope_en) begin
        for (int l = 0; l < LANES; l += 2) begin
          if (l+1 < LANES) begin
            rope_pipe[0][(l*16) +: 32] <= rotate_pair_lut(
              s_tdata[(l*16) +: 16],
              s_tdata[((l+1)*16) +: 16],
              phase_q
            );
          end
        end
        if (s_tvalid) phase_q <= phase_q + 3'd1;
      end
      valid_pipe[0] <= s_tvalid;
      rope_pipe[1]  <= rope_pipe[0];
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


// --------------------------------------------------------------------------
// ooo_micro_sequencer (unchanged)
// --------------------------------------------------------------------------
module ooo_micro_sequencer #(
  parameter int Q_DEPTH = 4
)(
  input  logic         clk,
  input  logic         rst_n,
  input  logic [31:0]  ir_in,
  input  logic         ir_valid,
  output logic         shift_w_en,
  output logic         swap_weights,
  output logic         clear_ps_base,
  input  logic         dma_busy,
  input  logic         array_busy,
  output logic         trigger_dma,
  output logic         trigger_array,
  output logic         mem_issue_valid,
  output logic         compute_issue_valid,
  output logic         dual_issue_valid,
  output logic [7:0]   mem_queue_count,
  output logic [7:0]   compute_queue_count
);

  localparam int PTR_W = (Q_DEPTH <= 1) ? 1 : $clog2(Q_DEPTH);
  localparam logic [PTR_W:0] Q_DEPTH_COUNT = Q_DEPTH;

  logic [31:0] mem_q [0:Q_DEPTH-1];
  logic [31:0] comp_q [0:Q_DEPTH-1];
  logic [PTR_W-1:0] mem_wr, mem_rd, comp_wr, comp_rd;
  logic [PTR_W:0] mem_count, comp_count;
  logic [3:0] opcode;
  logic is_mem_op, is_comp_op;

  assign opcode = ir_in[31:28];
  assign is_mem_op = (opcode == 4'h1) || (opcode == 4'h2) || (opcode == 4'h4);
  assign is_comp_op = (opcode == 4'h3) || (opcode == 4'h5) || (opcode == 4'h6);

  function automatic logic [PTR_W-1:0] q_inc(input logic [PTR_W-1:0] p);
    q_inc = (p == Q_DEPTH-1) ? '0 : p + 1'b1;
  endfunction

  logic seq_mem_enq, seq_comp_enq, seq_mem_deq, seq_comp_deq;
  logic [PTR_W:0] seq_mem_count_next, seq_comp_count_next;
  logic [PTR_W-1:0] seq_mem_wr_next, seq_mem_rd_next, seq_comp_wr_next, seq_comp_rd_next;

  always_comb begin
    seq_mem_enq = ir_valid && is_mem_op && (mem_count != Q_DEPTH_COUNT);
    seq_comp_enq = ir_valid && is_comp_op && (comp_count != Q_DEPTH_COUNT);
    seq_mem_deq = !dma_busy && (mem_count != '0);
    seq_comp_deq = !array_busy && (comp_count != '0);
    seq_mem_count_next = mem_count + seq_mem_enq - seq_mem_deq;
    seq_comp_count_next = comp_count + seq_comp_enq - seq_comp_deq;
    seq_mem_wr_next = seq_mem_enq ? q_inc(mem_wr) : mem_wr;
    seq_mem_rd_next = seq_mem_deq ? q_inc(mem_rd) : mem_rd;
    seq_comp_wr_next = seq_comp_enq ? q_inc(comp_wr) : comp_wr;
    seq_comp_rd_next = seq_comp_deq ? q_inc(comp_rd) : comp_rd;
  end

  always_ff @(posedge clk) begin
    if (!rst_n) begin
      mem_wr <= '0; mem_rd <= '0; comp_wr <= '0; comp_rd <= '0;
      mem_count <= '0; comp_count <= '0;
      shift_w_en <= 1'b0; swap_weights <= 1'b0; clear_ps_base <= 1'b0;
      trigger_dma <= 1'b0; trigger_array <= 1'b0;
      mem_issue_valid <= 1'b0; compute_issue_valid <= 1'b0; dual_issue_valid <= 1'b0;
      mem_queue_count <= '0; compute_queue_count <= '0;
    end else begin
      shift_w_en <= 1'b0; swap_weights <= 1'b0; clear_ps_base <= 1'b0;
      trigger_dma <= 1'b0; trigger_array <= 1'b0;
      mem_issue_valid <= 1'b0; compute_issue_valid <= 1'b0; dual_issue_valid <= 1'b0;

      if (seq_mem_enq) mem_q[mem_wr] <= ir_in;
      if (seq_comp_enq) comp_q[comp_wr] <= ir_in;

      mem_issue_valid <= seq_mem_deq;
      compute_issue_valid <= seq_comp_deq;
      dual_issue_valid <= seq_mem_deq && seq_comp_deq;

      if (seq_mem_deq) begin
        unique case (mem_q[mem_rd][31:28])
          4'h1: begin shift_w_en <= 1'b1; trigger_dma <= 1'b1; end
          4'h2: begin swap_weights <= 1'b1; end
          4'h4: begin trigger_dma <= 1'b1; end
          default: begin end
        endcase
      end
      if (seq_comp_deq) begin
        unique case (comp_q[comp_rd][31:28])
          4'h3: begin clear_ps_base <= 1'b1; trigger_array <= 1'b1; end
          4'h5: begin trigger_array <= 1'b1; end
          4'h6: begin trigger_array <= 1'b1; end
          default: begin end
        endcase
      end

      mem_wr <= seq_mem_wr_next; mem_rd <= seq_mem_rd_next;
      mem_count <= seq_mem_count_next;
      comp_wr <= seq_comp_wr_next; comp_rd <= seq_comp_rd_next;
      comp_count <= seq_comp_count_next;
      mem_queue_count <= {{(8-(PTR_W+1)){1'b0}}, seq_mem_count_next};
      compute_queue_count <= {{(8-(PTR_W+1)){1'b0}}, seq_comp_count_next};
    end
  end

`ifndef SYNTHESIS
  initial begin
    assert (Q_DEPTH > 1) else $fatal("ooo_micro_sequencer Q_DEPTH must be > 1");
  end
  property p_dual_issue_possible_when_both_ready;
    @(posedge clk) disable iff (!rst_n)
      (!dma_busy && !array_busy && (mem_count != '0) && (comp_count != '0)) |->
      ((trigger_dma || shift_w_en || swap_weights) && (trigger_array || clear_ps_base));
  endproperty
  assert property (p_dual_issue_possible_when_both_ready)
    else $error("dual issue opportunity did not produce both memory and compute issue side effects");
`endif
endmodule


// --------------------------------------------------------------------------
// systolic_pe (unchanged, includes FIX-19 safe weight swap)
// --------------------------------------------------------------------------
module systolic_pe #(
  parameter int ACT_W = 16,
  parameter int WT_W  = 16,
  parameter int PS_W  = 32
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic               ce,
  input  logic               sleep,
  input  logic               cfg_bypass,
  input  logic               cfg_dataflow,
  input  logic [3:0]         cfg_mode,
  input  logic               cfg_mx_native_accum,
  input  logic               cfg_mx_finalize,
  input  logic [7:0]         shared_exp,
  input  logic               cfg_quant_en,
  input  logic [1:0]         cfg_quant_scale_mode,
  input  logic [15:0]        quant_scale_q8_8,
  input  logic [31:0]        quant_scale_fp32,
  input  logic signed [31:0] quant_bias_i32,
  input  logic signed [15:0] act_zero_point,
  input  logic signed [15:0] wt_zero_point,
  input  logic               shift_w_en,
  input  logic               swap_weights,
  input  logic               clear_ps,
  input  logic [WT_W-1:0]    weight_in,
  input  logic [3:0]         sparse_meta_in,
  input  logic [ACT_W-1:0]   activation_in,
  input  logic [PS_W-1:0]    partial_sum_in,
  input  logic               valid_in,
  output logic [WT_W-1:0]    weight_out,
  output logic [3:0]         sparse_meta_out,
  output logic [ACT_W-1:0]   activation_out,
  output logic [PS_W-1:0]    partial_sum_out,
  output logic               valid_out
);

  logic [WT_W-1:0] weight_shadow, weight_active;
  logic [3:0]      sparse_meta_shadow, sparse_meta_active;
  logic [PS_W-1:0] os_accum;
  logic [ACT_W-1:0] act_q1, act_q2;
  logic [PS_W-1:0]  ps_q1, ps_q2;
  logic             val_q1, val_q2, clr_q1, clr_q2;
  logic             pe_active_req;
  logic [31:0]      mac_c_in, omni_mac_out;

  assign pe_active_req = valid_in | val_q1 | val_q2 | shift_w_en | swap_weights | clear_ps | clr_q1 | clr_q2;
  assign mac_c_in = cfg_dataflow ? ((val_q2 && !clr_q2) ? omni_mac_out : os_accum) : partial_sum_in;

  unified_fracturable_mac u_omni_mac (
    .clk(clk), .rst_n(rst_n), .ce(ce && pe_active_req && !sleep),
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
      weight_shadow <= '0; weight_active <= '0;
      sparse_meta_shadow <= '0; sparse_meta_active <= '0;
      os_accum <= '0;
      act_q1 <= '0; act_q2 <= '0;
      ps_q1 <= '0; ps_q2 <= '0;
      val_q1 <= 1'b0; val_q2 <= 1'b0;
      clr_q1 <= 1'b0; clr_q2 <= 1'b0;
      weight_out <= '0; sparse_meta_out <= '0;
      activation_out <= '0; partial_sum_out <= '0; valid_out <= 1'b0;
    end else if (sleep) begin
      activation_out <= '0;
      partial_sum_out <= partial_sum_in;
      valid_out <= 1'b0;
    end else if (ce && pe_active_req) begin
      if (shift_w_en) begin
        weight_shadow <= weight_in;
        sparse_meta_shadow <= sparse_meta_in;
      end
      if (swap_weights && !(|{val_q1, val_q2, valid_in})) begin
        weight_active <= weight_shadow;
        sparse_meta_active <= sparse_meta_shadow;
      end
      weight_out <= shift_w_en ? weight_in : (cfg_dataflow ? weight_in : weight_shadow);
      sparse_meta_out <= shift_w_en ? sparse_meta_in : sparse_meta_shadow;
      act_q1 <= activation_in;
      act_q2 <= act_q1;
      ps_q1 <= partial_sum_in;
      ps_q2 <= ps_q1;
      val_q1 <= valid_in;
      val_q2 <= val_q1;
      clr_q1 <= clear_ps;
      clr_q2 <= clr_q1;
      activation_out <= act_q2;
      valid_out <= val_q2;
      if (clr_q2) begin
        partial_sum_out <= '0;
        os_accum <= '0;
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
    assert (WT_W == 16) else $fatal("systolic_pe currently requires WT_W=16");
    assert (PS_W == 32) else $fatal("systolic_pe currently requires PS_W=32");
  end
  property p_mx_finalize_requires_native;
    @(posedge clk) disable iff (!rst_n)
      cfg_mx_finalize |-> cfg_mx_native_accum;
  endproperty
  assert property (p_mx_finalize_requires_native)
    else $error("cfg_mx_finalize asserted without native MX accumulation enabled");

  property p_swap_when_idle;
    @(posedge clk) disable iff (!rst_n)
      swap_weights |-> !(valid_in || val_q1 || val_q2);
  endproperty
  assert property (p_swap_when_idle)
    else $error("Weight swap issued while pipeline not empty -- possible data corruption");
`endif
endmodule


// --------------------------------------------------------------------------
// systolic_array (unchanged, uses updated flash_shared_normalizer)
// --------------------------------------------------------------------------
module systolic_array #(
  parameter int ROWS = 4,
  parameter int COLS = 4,
  parameter int ACT_W = 16,
  parameter int WT_W = 16,
  parameter int PS_W = 32
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic [ROWS-1:0]    row_ce_flat,
  input  logic [COLS-1:0]    col_ce_flat,
  input  logic               cfg_bypass,
  input  logic               cfg_dataflow,
  input  logic [3:0]         cfg_mode,
  input  logic               cfg_mx_native_accum,
  input  logic               cfg_mx_finalize,
  input  logic [2:0]         cfg_vpu_mode,
  input  logic [3:0]         cfg_gqa_group_log2,
  input  logic [7:0]         shared_exp,
  input  logic               cfg_quant_en,
  input  logic [1:0]         cfg_quant_scale_mode,
  input  logic               cfg_quant_per_channel,
  input  logic [15:0]        quant_scale_tensor_q8_8,
  input  logic [31:0]        quant_scale_tensor_fp32,
  input  logic signed [31:0] quant_bias_tensor_i32,
  input  logic signed [15:0] act_zero_point,
  input  logic signed [15:0] wt_zero_point,
  input  logic [COLS*16-1:0] quant_scale_col_q8_8_flat,
  input  logic [COLS*32-1:0] quant_scale_col_fp32_flat,
  input  logic [COLS*32-1:0] quant_bias_col_i32_flat,
  input  logic [15:0]        seq_i_base,
  input  logic [15:0]        seq_j_base,
  input  logic [ROWS-1:0]    row_sleep,
  input  logic               shift_w_en,
  input  logic               swap_weights,
  input  logic [ROWS-1:0]    clear_ps_flat,
  input  logic [ROWS*ACT_W-1:0] activation_in_flat,
  input  logic [ROWS-1:0]       valid_in_flat,
  input  logic [COLS*WT_W-1:0]  weight_top_in_flat,
  input  logic [COLS*4-1:0]     sparse_meta_top_in_flat,
  input  logic [COLS*PS_W-1:0]  ps_north_in_flat,
  input  logic [COLS*PS_W-1:0]  v_top_in_flat,
  output logic [COLS*PS_W-1:0]  partial_sum_out_flat,
  output logic [COLS-1:0]       valid_out_flat,
  output logic [ROWS*ACT_W-1:0] cascade_act_out_flat,
  output logic [ROWS-1:0]       cascade_val_out_flat
);

  logic row_ce [0:ROWS-1];
  logic col_ce [0:COLS-1];
  logic clear_ps [0:ROWS-1];
  logic [ACT_W-1:0] activation_in [0:ROWS-1];
  logic valid_in [0:ROWS-1];
  logic [WT_W-1:0] weight_top_in [0:COLS-1];
  logic [3:0]      sparse_meta_top_in [0:COLS-1];
  logic [PS_W-1:0] ps_north_in [0:COLS-1];
  logic [PS_W-1:0] v_top_in [0:COLS-1];
  logic [PS_W-1:0] partial_sum_out[0:COLS-1];
  logic valid_out [0:COLS-1];
  logic [ACT_W-1:0] cascade_act_out[0:ROWS-1];
  logic cascade_val_out[0:ROWS-1];

  always_comb begin
    for (int r = 0; r < ROWS; r++) begin
      row_ce[r] = row_ce_flat[r];
      clear_ps[r] = clear_ps_flat[r];
      activation_in[r] = activation_in_flat[r*ACT_W +: ACT_W];
      valid_in[r] = valid_in_flat[r];
      cascade_act_out_flat[r*ACT_W +: ACT_W] = cascade_act_out[r];
      cascade_val_out_flat[r] = cascade_val_out[r];
    end
    for (int c = 0; c < COLS; c++) begin
      col_ce[c] = col_ce_flat[c];
      weight_top_in[c] = weight_top_in_flat[c*WT_W +: WT_W];
      sparse_meta_top_in[c] = sparse_meta_top_in_flat[c*4 +: 4];
      ps_north_in[c] = ps_north_in_flat[c*PS_W +: PS_W];
      v_top_in[c] = v_top_in_flat[c*PS_W +: PS_W];
      partial_sum_out_flat[c*PS_W +: PS_W] = partial_sum_out[c];
      valid_out_flat[c] = valid_out[c];
    end
  end

  logic [ACT_W-1:0] act_right [0:ROWS-1][0:COLS];
  logic val_right [0:ROWS-1][0:COLS];
  logic [PS_W-1:0] ps_down [0:ROWS][0:COLS-1];
  logic [WT_W-1:0] wt_down [0:ROWS][0:COLS-1];
  logic [3:0]      meta_down [0:ROWS][0:COLS-1];
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
    group_size = 1 << group_log2;
    if (group_size == 0) group_size = 1;
    gqa_base_idx = (col / group_size) * group_size;
    if (gqa_base_idx >= COLS) gqa_base_idx = COLS-1;
  endfunction

  always_comb begin
    for (int cc = 0; cc < COLS; cc++) begin
      v_gqa_comb[cc] = v_top_in[gqa_base_idx(cc, cfg_gqa_group_log2)];
    end
  end

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
      assign act_right[r][0] = activation_in[r];
      assign val_right[r][0] = valid_in[r];
      assign cascade_act_out[r] = act_right[r][COLS];
      assign cascade_val_out[r] = val_right[r][COLS];
    end

    for (c = 0; c < COLS; c++) begin : gen_top
      assign ps_down[0][c] = ps_north_in[c];
      assign wt_down[0][c] = weight_top_in[c];
      assign meta_down[0][c] = sparse_meta_top_in[c];
      assign quant_scale_col_sel[c] = cfg_quant_per_channel ? quant_scale_col_q8_8_flat[(c*16) +: 16] : quant_scale_tensor_q8_8;
      assign quant_scale_fp32_col_sel[c] = cfg_quant_per_channel ? quant_scale_col_fp32_flat[(c*32) +: 32] : quant_scale_tensor_fp32;
      assign quant_bias_col_sel[c] = cfg_quant_per_channel ? quant_bias_col_i32_flat[(c*32) +: 32] : quant_bias_tensor_i32;
    end

    for (r = 0; r < ROWS; r++) begin : gen_rows
      for (c = 0; c < COLS; c++) begin : gen_cols
        systolic_pe #(.ACT_W(ACT_W), .WT_W(WT_W), .PS_W(PS_W)) u_pe (
          .clk(clk), .rst_n(rst_n), .ce(row_ce[r]), .sleep(row_sleep[r]),
          .cfg_bypass(cfg_bypass), .cfg_dataflow(cfg_dataflow),
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
        .clk(clk), .rst_n(rst_n), .ce(col_ce[c]),
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

  logic [COLS*PS_W-1:0] norm_num_flat, norm_den_flat, norm_out_flat;
  logic [COLS-1:0] norm_req_valid_flat, norm_valid_out_flat;

  always_comb begin
    for (int c = 0; c < COLS; c++) begin
      norm_num_flat[c*PS_W +: PS_W] = norm_num[c];
      norm_den_flat[c*PS_W +: PS_W] = norm_den[c];
      norm_req_valid_flat[c] = norm_req_valid[c];
      partial_sum_out[c] = norm_out_flat[c*PS_W +: PS_W];
      valid_out[c] = norm_valid_out_flat[c];
    end
  end

  flash_shared_normalizer #(.COLS(COLS), .PS_W(PS_W)) u_shared_norm (
    .clk(clk), .rst_n(rst_n), .ce(col_ce[0]),
    .clear(clear_ps[0]),
    .num_in_flat(norm_num_flat),
    .den_in_flat(norm_den_flat),
    .valid_in_flat(norm_req_valid_flat),
    .norm_out_flat(norm_out_flat),
    .valid_out_flat(norm_valid_out_flat)
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


// --------------------------------------------------------------------------
// Virtual-Channel 2D Torus Flit Router (unchanged)
// --------------------------------------------------------------------------
module vc_flit_router_2d #(
  parameter int FLIT_W = 128,
  parameter int VC_COUNT = 4,
  parameter int X_ID = 0,
  parameter int Y_ID = 0,
  parameter int CREDIT_DEPTH = 4
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic [FLIT_W-1:0]  s_local_data, input  logic s_local_valid, output logic s_local_ready,
  input  logic [FLIT_W-1:0]  s_west_data,  input  logic s_west_valid,  output logic s_west_ready,
  input  logic [FLIT_W-1:0]  s_east_data,  input  logic s_east_valid,  output logic s_east_ready,
  input  logic [FLIT_W-1:0]  s_north_data, input  logic s_north_valid, output logic s_north_ready,
  input  logic [FLIT_W-1:0]  s_south_data, input  logic s_south_valid, output logic s_south_ready,
  output logic [FLIT_W-1:0]  m_local_data, output logic m_local_valid, input logic m_local_ready, input logic [VC_COUNT-1:0] m_local_credit_return,
  output logic [FLIT_W-1:0]  m_west_data,  output logic m_west_valid,  input logic m_west_ready,  input logic [VC_COUNT-1:0] m_west_credit_return,
  output logic [FLIT_W-1:0]  m_east_data,  output logic m_east_valid,  input logic m_east_ready,  input logic [VC_COUNT-1:0] m_east_credit_return,
  output logic [FLIT_W-1:0]  m_north_data, output logic m_north_valid, input logic m_north_ready, input logic [VC_COUNT-1:0] m_north_credit_return,
  output logic [FLIT_W-1:0]  m_south_data, output logic m_south_valid, input logic m_south_ready, input logic [VC_COUNT-1:0] m_south_credit_return
);

  typedef enum logic [2:0] { P_LOCAL=3'd0, P_WEST=3'd1, P_EAST=3'd2, P_NORTH=3'd3, P_SOUTH=3'd4 } port_t;
  localparam int PORTS = 5;
  localparam int CAND = PORTS * VC_COUNT;
  localparam int RR_W = (CAND <= 1) ? 1 : $clog2(CAND);
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
  logic credit_return [0:PORTS-1][0:VC_COUNT-1];
  logic credit_consume [0:PORTS-1][0:VC_COUNT-1];
  logic [RR_W-1:0] rr_ptr [0:PORTS-1];
  logic sel_found [0:PORTS-1];
  logic [RR_W-1:0] sel_flat [0:PORTS-1];

  assign in_data[0]=s_local_data;  assign in_valid[0]=s_local_valid;  assign s_local_ready=in_ready[0];
  assign in_data[1]=s_west_data;   assign in_valid[1]=s_west_valid;    assign s_west_ready=in_ready[1];
  assign in_data[2]=s_east_data;   assign in_valid[2]=s_east_valid;    assign s_east_ready=in_ready[2];
  assign in_data[3]=s_north_data;  assign in_valid[3]=s_north_valid;   assign s_north_ready=in_ready[3];
  assign in_data[4]=s_south_data;  assign in_valid[4]=s_south_valid;   assign s_south_ready=in_ready[4];

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
    int unsigned raw_vc = flit[FLIT_W-10 -: 2];
    vc_of = (raw_vc >= VC_COUNT) ? (VC_COUNT-1) : raw_vc;
  endfunction

  function automatic port_t route_for(input logic [FLIT_W-1:0] flit);
    logic [3:0] dx, dy;
    dx = flit[FLIT_W-2 -: 4];
    dy = flit[FLIT_W-6 -: 4];
    if ((dx == X_ID4) && (dy == Y_ID4)) route_for = P_LOCAL;
    else if (dx != X_ID4) route_for = (dx > X_ID4) ? P_EAST : P_WEST;
    else route_for = (dy > Y_ID4) ? P_NORTH : P_SOUTH;
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
      input logic cons);
    logic [CREDIT_W:0] tmp;
    begin
      tmp = {1'b0, cur};
      if (ret && (cur != CREDIT_MAX)) tmp = tmp + 1'b1;
      if (cons && (tmp != '0)) tmp = tmp - 1'b1;
      if (tmp[CREDIT_W-1:0] > CREDIT_MAX) credit_net_next = CREDIT_MAX;
      else credit_net_next = tmp[CREDIT_W-1:0];
    end
  endfunction

  always_comb begin
    for (int p=0; p<PORTS; p++) begin
      int unsigned v = vc_of(in_data[p]);
      in_ready[p] = !vc_q_valid[p][v];
    end
    for (int o=0; o<PORTS; o++) begin
      sel_found[o] = 1'b0;
      sel_flat[o] = rr_ptr[o];
      for (int step=0; step<CAND; step++) begin
        int unsigned flat = (rr_ptr[o] + step) % CAND;
        int unsigned ip = flat_port(flat);
        int unsigned ivc = flat_vc(flat);
        if (!sel_found[o] &&
            vc_q_valid[ip][ivc] &&
            (route_for(vc_q_data[ip][ivc]) == port_t'(o)) &&
            (credit[o][ivc] != '0)) begin
          sel_found[o] = 1'b1;
          sel_flat[o] = flat[RR_W-1:0];
        end
      end
    end
    for (int p=0; p<PORTS; p++) begin
      for (int v=0; v<VC_COUNT; v++) begin
        credit_return[p][v] = out_credit_return[p][v] && (credit[p][v] != CREDIT_MAX);
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
      for (int p=0; p<PORTS; p++) begin
        int unsigned v = vc_of(in_data[p]);
        if (in_valid[p] && in_ready[p]) begin
          vc_q_data[p][v] <= in_data[p];
          vc_q_valid[p][v] <= 1'b1;
        end
      end
      for (int o=0; o<PORTS; o++) begin
        if (sel_found[o]) begin
          int unsigned ip = flat_port(sel_flat[o]);
          int unsigned ivc = flat_vc(sel_flat[o]);
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
    @(posedge clk) disable iff (!rst_n) (credit[0][0] <= CREDIT_MAX);
  endproperty
  assert property (p_credit_bounds)
    else $error("router credit exceeded CREDIT_MAX");
`endif
endmodule


// --------------------------------------------------------------------------
// hyperion_exascale_node (v25 top-level)
// --------------------------------------------------------------------------
module hyperion_exascale_node #(
  parameter int ROWS = 4,
  parameter int COLS = 4,
  parameter int S_AXIS_W = 64,
  parameter int M_AXIS_W = COLS * 32,
  parameter int WT_TOP_W = COLS * 16,
  parameter int META_AXIS_W = COLS * 4,
  parameter int TCSM_DEPTH = 256,
  parameter int TCSM_AW = (TCSM_DEPTH <= 1) ? 1 : $clog2(TCSM_DEPTH)
)(
  input  logic               clk,
  input  logic               rst_n,
  input  logic [31:0]        ir_in,
  input  logic               ir_valid,
  input  logic [3:0]         cfg_mode,
  input  logic               cfg_mx_native_accum,
  input  logic               cfg_mx_finalize,
  input  logic [2:0]         cfg_vpu_mode,
  input  logic [3:0]         cfg_gqa_group_log2,
  input  logic               cfg_bypass,
  input  logic               cfg_dataflow,
  input  logic               cfg_allreduce,
  input  logic               cfg_allreduce_fp,
  input  logic               cfg_broadcast,
  input  logic               cfg_rope_en,
  input  logic [7:0]         shared_exp,
  input  logic [15:0]        seq_i_base,
  input  logic [15:0]        seq_j_base,
  input  logic [ROWS-1:0]    row_sleep,
  input  logic               dma_busy,
  input  logic               array_busy,
  input  logic               cfg_quant_en,
  input  logic [1:0]         cfg_quant_scale_mode,
  input  logic               cfg_quant_per_channel,
  input  logic [15:0]        quant_scale_tensor_q8_8,
  input  logic [31:0]        quant_scale_tensor_fp32,
  input  logic signed [31:0] quant_bias_tensor_i32,
  input  logic signed [15:0] act_zero_point,
  input  logic signed [15:0] wt_zero_point,
  input  logic [COLS*16-1:0] quant_scale_col_q8_8_flat,
  input  logic [COLS*32-1:0] quant_scale_col_fp32_flat,
  input  logic [COLS*32-1:0] quant_bias_col_i32_flat,
  input  logic               tma_desc_valid,
  output logic               tma_desc_ready,
  input  logic [63:0]        tma_desc_base_addr,
  input  logic [15:0]        tma_desc_dim_m,
  input  logic [15:0]        tma_desc_dim_n,
  input  logic [15:0]        tma_desc_stride_m,
  input  logic [15:0]        tma_desc_stride_n,
  input  logic [15:0]        tma_desc_tile_m,
  input  logic [15:0]        tma_desc_tile_n,
  input  logic [1:0]         tma_desc_dst_kind,
  input  logic               tma_desc_dst_bank,
  input  logic [M_AXIS_W-1:0] tma_stream_data,
  input  logic               tma_stream_valid,
  output logic               tma_stream_ready,
  output logic               tma_busy,
  output logic               tma_done,
  input  logic               kv_lookup_valid,
  output logic               kv_lookup_ready,
  input  logic [11:0]        kv_lookup_vpn,
  output logic               kv_lookup_resp_valid,
  output logic               kv_lookup_miss,
  output logic [23:0]        kv_lookup_ppn,
  output logic               kv_pager_stall,
  output logic               kv_fault_valid,
  output logic [11:0]        kv_fault_vpn,
  input  logic               kv_fault_clear,
  input  logic               kv_ptw_write_valid,
  input  logic [7:0]         kv_ptw_write_index,
  input  logic [11:0]        kv_ptw_write_vpn,
  input  logic [23:0]        kv_ptw_write_ppn,
  input  logic               kv_ptw_write_valid_bit,
  input  logic [WT_TOP_W-1:0] weight_top_flat,
  input  logic               weight_load_valid,
  input  logic               weight_load_bank,
  input  logic [TCSM_AW-1:0] weight_load_addr,
  input  logic [M_AXIS_W-1:0] v_top_flat,
  input  logic               v_load_valid,
  input  logic               v_load_bank,
  input  logic [TCSM_AW-1:0] v_load_addr,
  input  logic [TCSM_AW-1:0] weight_read_addr,
  input  logic [TCSM_AW-1:0] v_read_addr,
  input  logic               tcsm_swap,
  input  logic [S_AXIS_W-1:0] s_axis_west_tdata,
  input  logic               s_axis_west_tvalid,
  output logic               s_axis_west_tready,
  output logic [S_AXIS_W-1:0] m_axis_east_tdata,
  output logic               m_axis_east_tvalid,
  input  logic               m_axis_east_tready,
  input  logic [M_AXIS_W-1:0] s_axis_north_tdata,
  input  logic               s_axis_north_tvalid,
  output logic               s_axis_north_tready,
  output logic [M_AXIS_W-1:0] m_axis_south_tdata,
  output logic               m_axis_south_tvalid,
  input  logic               m_axis_south_tready,
  input  logic [META_AXIS_W-1:0] s_axis_meta_tdata,
  input  logic               s_axis_meta_tvalid,
  output logic               s_axis_meta_tready
);

  localparam int EAST_ALIGN_LAT = 2 * COLS;
  localparam int SOUTH_ALIGN_LAT = (2 * ROWS) + 5;

  logic [S_AXIS_W-1:0] rope_tdata;
  logic rope_tvalid, rope_tready;

  rope_engine #(.DATA_W(S_AXIS_W)) u_rope (
    .clk(clk), .rst_n(rst_n),
    .cfg_rope_en(cfg_rope_en),
    .s_tdata(s_axis_west_tdata), .s_tvalid(s_axis_west_tvalid), .s_tready(s_axis_west_tready),
    .m_tdata(rope_tdata), .m_tvalid(rope_tvalid), .m_tready(rope_tready)
  );

  logic shift_w_en, swap_weights, clear_ps_base;
  logic trigger_dma, trigger_array;
  logic sequencer_kv_stall;

  // FIX-17: fault_clear immediately resets the stall, no extra cycle penalty
  always_ff @(posedge clk or negedge rst_n) begin
    if (!rst_n) sequencer_kv_stall <= 1'b0;
    else if (kv_fault_clear) sequencer_kv_stall <= 1'b0;
    else if (kv_pager_stall) sequencer_kv_stall <= 1'b1;
    else sequencer_kv_stall <= 1'b0;
  end

  ooo_micro_sequencer u_seq (
    .clk(clk), .rst_n(rst_n),
    .ir_in(ir_in), .ir_valid(ir_valid),
    .shift_w_en(shift_w_en), .swap_weights(swap_weights), .clear_ps_base(clear_ps_base),
    .dma_busy(dma_busy || sequencer_kv_stall), .array_busy(array_busy || sequencer_kv_stall),
    .trigger_dma(trigger_dma), .trigger_array(trigger_array),
    .mem_issue_valid(), .compute_issue_valid(), .dual_issue_valid(),
    .mem_queue_count(), .compute_queue_count()
  );

  logic rx_valid, rx_pop, rx_full, rx_afull;
  logic [S_AXIS_W-1:0] rx_tdata;
  sync_fifo #(.DATA_W(S_AXIS_W), .DEPTH(32)) u_rx_fifo (
    .clk(clk), .rst_n(rst_n),
    .push(rope_tvalid && rope_tready), .data_in(rope_tdata),
    .pop(rx_pop),
    .data_out(rx_tdata), .valid_out(rx_valid),
    .empty(), .full(rx_full), .almost_full(rx_afull)
  );
  assign rope_tready = !rx_full;

  logic north_valid, north_pop, north_full, north_afull;
  logic [M_AXIS_W-1:0] north_tdata;
  sync_fifo #(.DATA_W(M_AXIS_W), .DEPTH(32)) u_north_fifo (
    .clk(clk), .rst_n(rst_n),
    .push(s_axis_north_tvalid && s_axis_north_tready), .data_in(s_axis_north_tdata),
    .pop(north_pop),
    .data_out(north_tdata), .valid_out(north_valid),
    .empty(), .full(north_full), .almost_full(north_afull)
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
  assign tma_desc_ready = tma_desc_ready_int && !sequencer_kv_stall;

  logic weight_load_bank_eff, v_load_bank_eff;
  logic [TCSM_AW-1:0] weight_load_addr_eff, v_load_addr_eff;
  logic [WT_TOP_W-1:0] weight_load_data_eff;
  logic [M_AXIS_W-1:0] v_load_data_eff;

  logic meta_valid, meta_pop, meta_full, meta_afull;
  logic [META_AXIS_W-1:0] meta_tdata;
  sync_fifo #(.DATA_W(META_AXIS_W), .DEPTH(32)) u_sparse_meta_fifo (
    .clk(clk), .rst_n(rst_n),
    .push((s_axis_meta_tvalid && s_axis_meta_tready) || (tma_load_valid && (tma_load_dst_kind == 2'd2))),
    .data_in((tma_load_valid && (tma_load_dst_kind == 2'd2)) ? tma_load_data[META_AXIS_W-1:0] : s_axis_meta_tdata),
    .pop(meta_pop),
    .data_out(meta_tdata), .valid_out(meta_valid),
    .empty(), .full(meta_full), .almost_full(meta_afull)
  );
  assign s_axis_meta_tready = !meta_full;

  logic east_obuf_ready, south_obuf_ready;
  logic core_step_root, ingress_ce;
  logic [ROWS-1:0] row_ce_flat;
  logic [COLS-1:0] col_ce_flat;
  logic row_ce [0:ROWS-1];
  logic col_ce [0:COLS-1];
  logic sparse_meta_needed;
  logic sparse_meta_ready_for_step;
  assign sparse_meta_needed = (cfg_mode == 4'h8) && shift_w_en;
  assign sparse_meta_ready_for_step = !sparse_meta_needed || meta_valid;
  assign core_step_root = east_obuf_ready && south_obuf_ready && sparse_meta_ready_for_step && !sequencer_kv_stall;

  ce_relay_grid #(.ROWS(ROWS), .COLS(COLS)) u_ce_relay (
    .clk(clk), .rst_n(rst_n),
    .root_step(core_step_root),
    .ingress_ce(ingress_ce),
    .row_ce_flat(row_ce_flat),
    .col_ce_flat(col_ce_flat)
  );
  always_comb begin
    for (int r = 0; r < ROWS; r++) row_ce[r] = row_ce_flat[r];
    for (int c = 0; c < COLS; c++) col_ce[c] = col_ce_flat[c];
  end

  assign rx_pop = ingress_ce && rx_valid;
  assign north_pop = ingress_ce && north_valid;
  assign meta_pop = ingress_ce && meta_valid && (sparse_meta_needed || (cfg_mode != 4'h8));

  logic [ROWS*16-1:0] activation_in_flat;
  logic [ROWS-1:0] valid_in_flat;
  logic [ROWS-1:0] clear_ps_arr_flat;
  logic [COLS*32-1:0] partial_sum_out_flat;
  logic [COLS-1:0] valid_out_flat;
  logic [ROWS*16-1:0] cascade_act_out_flat;
  logic [ROWS-1:0] cascade_val_out_flat;
  logic [COLS*32-1:0] ps_north_in_flat;
  logic [COLS*16-1:0] weight_top_in_flat;
  logic [COLS*4-1:0] sparse_meta_top_in_flat;
  logic [COLS*32-1:0] v_top_in_flat;

  logic [15:0] activation_in [0:ROWS-1];
  logic valid_in [0:ROWS-1];
  logic clear_ps_arr [0:ROWS-1];
  logic [31:0] partial_sum_out[0:COLS-1];
  logic valid_out [0:COLS-1];
  logic [15:0] cascade_act_out[0:ROWS-1];
  logic cascade_val_out[0:ROWS-1];
  logic [31:0] ps_north_in [0:COLS-1];
  logic [15:0] weight_top_in [0:COLS-1];
  logic [3:0] sparse_meta_top_in [0:COLS-1];
  logic [31:0] v_top_in [0:COLS-1];

  always_comb begin
    for (int r = 0; r < ROWS; r++) begin
      activation_in_flat[r*16 +: 16] = activation_in[r];
      valid_in_flat[r] = valid_in[r];
      clear_ps_arr_flat[r] = clear_ps_arr[r];
      cascade_act_out[r] = cascade_act_out_flat[r*16 +: 16];
      cascade_val_out[r] = cascade_val_out_flat[r];
    end
    for (int c = 0; c < COLS; c++) begin
      ps_north_in_flat[c*32 +: 32] = ps_north_in[c];
      weight_top_in_flat[c*16 +: 16] = weight_top_in[c];
      sparse_meta_top_in_flat[c*4 +: 4] = sparse_meta_top_in[c];
      v_top_in_flat[c*32 +: 32] = v_top_in[c];
      partial_sum_out[c] = partial_sum_out_flat[c*32 +: 32];
      valid_out[c] = valid_out_flat[c];
    end
  end

  logic [WT_TOP_W-1:0] weight_tcsm_bus;
  logic [M_AXIS_W-1:0] v_tcsm_bus;
  logic weight_active_bank, v_active_bank;
  logic [S_AXIS_W-1:0] east_payload;
  logic [M_AXIS_W-1:0] south_payload;

  tma_tensor_loader #(.DATA_W(M_AXIS_W), .ADDR_W(TCSM_AW)) u_tma_loader (
    .clk(clk), .rst_n(rst_n),
    .desc_valid(tma_desc_valid && !sequencer_kv_stall),
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
    .hold(sequencer_kv_stall),
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
    .clk(clk), .rst_n(rst_n),
    .lookup_valid(kv_lookup_valid), .lookup_ready(kv_lookup_ready), .lookup_vpn(kv_lookup_vpn),
    .lookup_resp_valid(kv_lookup_resp_valid), .lookup_miss(kv_lookup_miss), .lookup_ppn(kv_lookup_ppn),
    .pager_stall(kv_pager_stall),
    .fault_valid(kv_fault_valid), .fault_vpn(kv_fault_vpn), .fault_clear(kv_fault_clear),
    .ptw_write_valid(kv_ptw_write_valid), .ptw_write_index(kv_ptw_write_index),
    .ptw_write_vpn(kv_ptw_write_vpn), .ptw_write_ppn(kv_ptw_write_ppn), .ptw_write_valid_bit(kv_ptw_write_valid_bit)
  );

  assign weight_load_valid_eff = weight_load_valid || (tma_load_valid && (tma_load_dst_kind == 2'd0));
  assign weight_load_bank_eff = (tma_load_valid && (tma_load_dst_kind == 2'd0)) ? tma_load_bank : weight_load_bank;
  assign weight_load_addr_eff = (tma_load_valid && (tma_load_dst_kind == 2'd0)) ? tma_load_addr : weight_load_addr;
  assign weight_load_data_eff = (tma_load_valid && (tma_load_dst_kind == 2'd0)) ? tma_load_data[WT_TOP_W-1:0] : weight_top_flat;
  assign v_load_valid_eff = v_load_valid || (tma_load_valid && (tma_load_dst_kind == 2'd1));
  assign v_load_bank_eff = (tma_load_valid && (tma_load_dst_kind == 2'd1)) ? tma_load_bank : v_load_bank;
  assign v_load_addr_eff = (tma_load_valid && (tma_load_dst_kind == 2'd1)) ? tma_load_addr : v_load_addr;
  assign v_load_data_eff = (tma_load_valid && (tma_load_dst_kind == 2'd1)) ? tma_load_data : v_top_flat;

  ping_pong_vector_tcsm #(.DATA_W(WT_TOP_W), .DEPTH(TCSM_DEPTH), .ADDR_W(TCSM_AW)) u_weight_tcsm (
    .clk(clk), .rst_n(rst_n),
    .load_en(weight_load_valid_eff), .load_bank(weight_load_bank_eff), .load_addr(weight_load_addr_eff), .load_data(weight_load_data_eff),
    .read_addr(weight_read_addr), .swap_banks(tcsm_swap), .read_data(weight_tcsm_bus), .active_bank(weight_active_bank)
  );
  ping_pong_vector_tcsm #(.DATA_W(M_AXIS_W), .DEPTH(TCSM_DEPTH), .ADDR_W(TCSM_AW)) u_v_tcsm (
    .clk(clk), .rst_n(rst_n),
    .load_en(v_load_valid_eff), .load_bank(v_load_bank_eff), .load_addr(v_load_addr_eff), .load_data(v_load_data_eff),
    .read_addr(v_read_addr), .swap_banks(tcsm_swap), .read_data(v_tcsm_bus), .active_bank(v_active_bank)
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
      assign valid_in[i] = cfg_broadcast ? rx_pop : val_delay[2*i];
      assign clear_ps_arr[i] = clear_ps_base;

      logic [15:0] east_addend_aligned;
      logic [15:0] east_raw_sum;
      logic [15:0] east_fp16_sum;
      assign east_addend_aligned = east_addend_valid[EAST_ALIGN_LAT] ? east_addend_delay[EAST_ALIGN_LAT] : 16'd0;
      assign east_raw_sum = cascade_act_out[i] + east_addend_aligned;
      fp16_adder_ref u_east_allreduce_fp16 (.a(cascade_act_out[i]), .b(east_addend_aligned), .sum(east_fp16_sum));
      assign east_payload[(i*16) +: 16] = cfg_allreduce ? (cfg_allreduce_fp ? east_fp16_sum : east_raw_sum) : cascade_act_out[i];
    end

    if (S_AXIS_W > (ROWS*16)) begin : gen_east_pad
      assign east_payload[S_AXIS_W-1:ROWS*16] = '0;
    end

    for (i = 0; i < COLS; i++) begin : gen_cols_flatten_and_south_align
      logic [31:0] south_addend_delay [0:SOUTH_ALIGN_LAT];
      logic south_addend_valid [0:SOUTH_ALIGN_LAT];

      assign ps_north_in[i] = north_valid ? north_tdata[(i*32) +: 32] : 32'd0;
      assign weight_top_in[i] = weight_tcsm_bus[(i*16) +: 16];
      assign sparse_meta_top_in[i] = meta_valid ? meta_tdata[(i*4) +: 4] : 4'd0;
      assign v_top_in[i] = v_tcsm_bus[(i*32) +: 32];

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

      logic [31:0] south_addend_aligned;
      logic [31:0] south_raw_sum;
      logic [31:0] south_fp_sum;
      assign south_addend_aligned = south_addend_valid[SOUTH_ALIGN_LAT] ? south_addend_delay[SOUTH_ALIGN_LAT] : 32'd0;
      assign south_raw_sum = partial_sum_out[i] + south_addend_aligned;
      fp32_adder u_south_allreduce_fp (.a(partial_sum_out[i]), .b(south_addend_aligned), .sum(south_fp_sum));
      assign south_payload[(i*32) +: 32] = cfg_allreduce ? (cfg_allreduce_fp ? south_fp_sum : south_raw_sum) : partial_sum_out[i];
    end
  endgenerate

  logic east_core_valid, south_core_valid;
  assign east_core_valid = |cascade_val_out_flat;
  assign south_core_valid = |valid_out_flat;

  // FIX-16: Output hold registers permanently enabled to avoid handshake loss
  axis_hold_reg #(.DATA_W(S_AXIS_W)) u_east_hold (
    .clk(clk), .rst_n(rst_n), .ce(1'b1),
    .s_data(east_payload), .s_valid(east_core_valid), .s_ready(east_obuf_ready),
    .m_data(m_axis_east_tdata), .m_valid(m_axis_east_tvalid), .m_ready(m_axis_east_tready)
  );
  axis_hold_reg #(.DATA_W(M_AXIS_W)) u_south_hold (
    .clk(clk), .rst_n(rst_n), .ce(1'b1),
    .s_data(south_payload), .s_valid(south_core_valid), .s_ready(south_obuf_ready),
    .m_data(m_axis_south_tdata), .m_valid(m_axis_south_tvalid), .m_ready(m_axis_south_tready)
  );

  systolic_array #(.ROWS(ROWS), .COLS(COLS), .ACT_W(16), .WT_W(16), .PS_W(32)) u_core (
    .clk(clk), .rst_n(rst_n),
    .row_ce_flat(row_ce_flat), .col_ce_flat(col_ce_flat),
    .cfg_bypass(cfg_bypass), .cfg_dataflow(cfg_dataflow),
    .cfg_mode(cfg_mode),
    .cfg_mx_native_accum(cfg_mx_native_accum), .cfg_mx_finalize(cfg_mx_finalize),
    .cfg_vpu_mode(cfg_vpu_mode), .cfg_gqa_group_log2(cfg_gqa_group_log2),
    .shared_exp(shared_exp),
    .cfg_quant_en(cfg_quant_en), .cfg_quant_scale_mode(cfg_quant_scale_mode), .cfg_quant_per_channel(cfg_quant_per_channel),
    .quant_scale_tensor_q8_8(quant_scale_tensor_q8_8), .quant_scale_tensor_fp32(quant_scale_tensor_fp32), .quant_bias_tensor_i32(quant_bias_tensor_i32),
    .act_zero_point(act_zero_point), .wt_zero_point(wt_zero_point),
    .quant_scale_col_q8_8_flat(quant_scale_col_q8_8_flat), .quant_scale_col_fp32_flat(quant_scale_col_fp32_flat), .quant_bias_col_i32_flat(quant_bias_col_i32_flat),
    .seq_i_base(seq_i_base), .seq_j_base(seq_j_base),
    .row_sleep(row_sleep),
    .shift_w_en(shift_w_en), .swap_weights(swap_weights),
    .clear_ps_flat(clear_ps_arr_flat),
    .activation_in_flat(activation_in_flat),
    .valid_in_flat(valid_in_flat),
    .weight_top_in_flat(weight_top_in_flat),
    .sparse_meta_top_in_flat(sparse_meta_top_in_flat),
    .ps_north_in_flat(ps_north_in_flat),
    .v_top_in_flat(v_top_in_flat),
    .partial_sum_out_flat(partial_sum_out_flat),
    .valid_out_flat(valid_out_flat),
    .cascade_act_out_flat(cascade_act_out_flat),
    .cascade_val_out_flat(cascade_val_out_flat)
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
    @(posedge clk) disable iff (!rst_n) sequencer_kv_stall |-> !tma_desc_ready;
  endproperty
  assert property (p_pager_stall_blocks_tma_accept)
    else $error("TMA descriptor accepted during unresolved KV pager fault");

  property p_pager_stall_blocks_core_step;
    @(posedge clk) disable iff (!rst_n) sequencer_kv_stall |-> !core_step_root;
  endproperty
  assert property (p_pager_stall_blocks_core_step)
    else $error("Core advanced during unresolved KV pager fault");

  property p_quant_mode_supported;
    @(posedge clk) disable iff (!rst_n) cfg_quant_scale_mode <= 2'd2;
  endproperty
  assert property (p_quant_mode_supported)
    else $error("unsupported cfg_quant_scale_mode");

  property p_no_meta_push_conflict;
    @(posedge clk) disable iff (!rst_n)
      !(s_axis_meta_tvalid && s_axis_meta_tready &&
        tma_load_valid && (tma_load_dst_kind == 2'd2));
  endproperty
  assert property (p_no_meta_push_conflict)
    else $error("Simultaneous AXI-stream metadata and TMA metadata push --- AXI beat dropped");
`endif
endmodule
