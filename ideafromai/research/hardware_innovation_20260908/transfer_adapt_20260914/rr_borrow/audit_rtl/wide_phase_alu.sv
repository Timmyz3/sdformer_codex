// Exactly eight 64-bit carry chains. There is no second adder in the consumer.
module wide_phase_alu(input logic split_fields,
 input logic [511:0] lhs,rhs,output logic [511:0] y);
 genvar l,b;
 generate for(l=0;l<8;l=l+1)begin:LANE
 for(b=0;b<64;b=b+1)begin:BIT
 wire cin;
 if(b==0)assign cin=1'b0;
 else if(b==13 || b==26 || b==39)assign cin=split_fields?1'b0:BIT[b-1].CARRY.cout;
 else assign cin=BIT[b-1].CARRY.cout;
 assign y[l*64+b]=lhs[l*64+b]^rhs[l*64+b]^cin;
 if(b<63)begin:CARRY
 wire cout;
 assign cout=(lhs[l*64+b]&rhs[l*64+b])|((lhs[l*64+b]^rhs[l*64+b])&cin);
 end
 end
 end endgenerate
endmodule
