#!/usr/bin/env python3
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parent;src=P.parent.parent/'round33_predictability/hardware/bitmap_supply.cpp'
s=src.read_text().replace('#include "bitmap_words.h"','static uint32_t bitmap_words[192]={};')
start=s.index('int main(){try{')
s=s[:start]+'''int main(int argc,char**argv){try{
 need(argc==2,"complete upgraded tile count required");int count=std::stoi(argv[1]);need(count>=151&&count<=910,"151..910 tiles");
 std::vector<int> order;for(int i=302;i<453;i++)order.push_back(i);for(int i=0;i<910;i++)if(i<302||i>=453)order.push_back(i);
 for(int j=0;j<count;j++)bitmap_words[order[j]/32]|=1u<<(order[j]%32);
 Geometry g={"patch.res0.conv0",240,320,96,96,true};
 std::cout<<"{\\\"schema\\\":\\\"r34_static_neighbor_operator_service_v1\\\",\\\"complete_Q8_tiles\\\":"<<count<<",\\\"mask_words\\\":[";
 for(int i=0;i<32;i++){if(i)std::cout<<",";std::cout<<bitmap_words[i];}
 std::cout<<"],\\\"result\\\":";Simulator(g,0,false).run();std::cout<<",\\\"AEE_measured\\\":false,\\\"whole_network\\\":false}\\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<"\\n";return 1;}}
'''
(P/'static_neighbor_supply.cpp').write_text(s)
(P/'STATIC_SOURCE_BINDINGS_V1.json').write_text(json.dumps({'original_source':str(src),'original_sha256':hashlib.sha256(src.read_bytes()).hexdigest(),'new_sha256':hashlib.sha256(s.encode()).hexdigest(),'changed':'only mask storage and main; original Simulator body unchanged','ordinary_mask_order':'fit-only static group2 IDs302..452, then smallest remaining complete tile IDs','AEE_measured':False},indent=2)+'\n')
