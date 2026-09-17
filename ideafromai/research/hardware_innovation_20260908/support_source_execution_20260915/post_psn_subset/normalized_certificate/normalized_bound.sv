module normalized_bound(
 input logic clk,group_enable,
 input logic positive,
 input logic signed[47:0] tau,nv,n_minus_one,p_minus_one,
 input logic[4:0] m,
 output logic signed[48:0] norm_n,norm_p,q_n,q_p,shift_n,shift_p,
 output logic lower_hit,upper_hit
);
 logic signed[48:0] add_a,add_n,add_p;
 logic carry_in;
 always_comb begin
  add_a=group_enable?49'($signed(tau)):49'($signed(nv));
  add_n=49'($signed(n_minus_one));add_p=49'($signed(p_minus_one));
  carry_in=group_enable?!positive:1'b1;
  // Two 49-bit adders with carry-in, shared between GROUP and PLANE.
  norm_n=add_a+add_n+49'(carry_in);
  norm_p=add_a+add_p+49'(carry_in);
  shift_n=$signed(q_n)>>>m;shift_p=$signed(q_p)>>>m;
  lower_hit=norm_n>shift_n;
  upper_hit=norm_p<=shift_p;
 end
 always_ff @(posedge clk)if(group_enable)begin q_n<=norm_n;q_p<=norm_p;end
endmodule
