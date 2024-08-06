#!/bin/bash

declare -A command_dict

path="/home/cell/ljy/SPEC2017_workspace" 
s_path="/home/cell/ljy/DVFS_RAY/powermodel/data/powerdata"


# command for benchmark key = dir, value = exe_cmd

# command_dict["549"]=
# command_dict["549"]=
# command_dict["549"]=
# command_dict["549"]=


# <intspeed refspeed>
# command_dict["600_1"]="./perlbench_s_base.mytest-m64 -I./lib checkspam.pl 2500 5 25 11 150 1 1 1 1 0<&- > checkspam.2500.5.25.11.150.1.1.1.1.out 2> checkspam.2500.5.25.11.150.1.1.1.1.err"
# command_dict["600_2"]="./perlbench_s_base.mytest-m64 -I./lib diffmail.pl 4 800 10 17 19 300 0<&- > diffmail.4.800.10.17.19.300.out 2> diffmail.4.800.10.17.19.300.err"
# command_dict["600_3"]="./perlbench_s_base.mytest-m64 -I./lib splitmail.pl 6400 12 26 16 100 0 0<&- > splitmail.6400.12.26.16.100.0.out 2> splitmail.6400.12.26.16.100.0.err"
# command_dict["602_1"]="./sgcc_base.mytest-m64 gcc-pp.c -O5 -fipa-pta -o gcc-pp.opts-O5_-fipa-pta.s 0<&- > gcc-pp.opts-O5_-fipa-pta.out 2> gcc-pp.opts-O5_-fipa-pta.err"
# command_dict["602_2"]="./sgcc_base.mytest-m64 gcc-pp.c -O5 -finline-limit=1000 -fselective-scheduling -fselective-scheduling2 -o gcc-pp.opts-O5_-finline-limit_1000_-fselective-scheduling_-fselective-scheduling2.s 0<&- > gcc-pp.opts-O5_-finline-limit_1000_-fselective-scheduling_-fselective-scheduling2.out 2> gcc-pp.opts-O5_-finline-limit_1000_-fselective-scheduling_-fselective-scheduling2.err"
# command_dict["602_3"]="./sgcc_base.mytest-m64 gcc-pp.c -O5 -finline-limit=24000 -fgcse -fgcse-las -fgcse-lm-fgcse-sm -o gcc-pp.opts-O5_-finline-limit_24000_-fgcse_-fgcse-las_-fgcse-lm_-fgcse-sm.s 0<&- > gcc-pp.opts-O5_-finline-limit_24000_-fgcse_-fgcse-las_-fgcse-lm_-fgcse-sm.out 2> gcc-pp.opts-O5_-finline-limit_24000_-fgcse_-fgcse-las_-fgcse-lm_-fgcse-sm.err"
# command_dict["605"]="./mcf_s_base.mytest-m64 inp.in 0<&- > inp.out 2> inp.err"
# command_dict["620"]="./omnetpp_s_base.mytest-m64 -c General -r 0 0<&- > omnetpp.General-0.out 2> omnetpp.General-0.err"
# command_dict["632"]="./xalancbmk_s_base.mytest-m64 -v t5.xml xalanc.xsl 0<&- > ref-t5.out 2> ref-t5.err"
# command_dict["625_1"]="./x264_s_base.mytest-m64 --pass 1 --stats x264_stats.log --bitrate 1000 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720 0<&- > run_000-1000_x264_s_base.icc_speed_no_affinity_x264_pass1.out 2> run_000-1000_x264_s_base.icc_speed_no_affinity_x264_pass1.err"
# command_dict["625_2"]="./x264_s_base.mytest-m64 --pass 2 --stats x264_stats.log --bitrate 1000 --dumpyuv 200 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720 0<&- > run_000-1000_x264_s_base.icc_speed_no_affinity_x264_pass2.out 2> run_000-1000_x264_s_base.icc_speed_no_affinity_x264_pass2.err"
# command_dict["625_3"]="./x264_s_base.mytest-m64 --seek 500 --dumpyuv 200 --frames 1250 -o BuckBunny_New.264 BuckBunny.yuv 1280x720 0<&- > run_0500-1250_x264_s_base.icc_speed_no_affinity_x264.out 2> run_0500-1250_x264_s_base.icc_speed_no_affinity_x264.err"
# command_dict["631"]="./deepsjeng_s_base.mytest-m64 ref.txt 0<&- > ref.out 2>> ref.err"
# command_dict["641"]="./leela_s_base.mytest-m64 ref.sgf 0<&- > ref.out 2>> ref.err"
# command_dict["648"]="./exchange2_s_base.mytest-m64 6 0<&- > exchange2.txt 2>> exchange2.err"
# command_dict["657_1"]="./xz_s_base.mytest-m64 cpu2006docs.tar.xz 6643 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 1036078272 1111795472 4 0<&- > cpu2006docs.tar-6643-4.out 2>> cpu2006docs.tar-6643-4.err"
# command_dict["657_2"]="./xz_s_base.mytest-m64 cld.tar.xz 1400 19cf30ae51eddcbefda78dd06014b4b96281456e078ca7c13e1c0c9e6aaea8dff3efb4ad6b0456697718cede6bd5454852652806a657bb56e07d61128434b474 536995164 539938872 8 0<&- > cld.tar-1400-8.out 2>> cld.tar-1400-8.err"


# <fpspeed refspeed>

# command_dict["603_1"]="./speed_bwaves_base.mytest-m64 bwaves_1 < bwaves_1.in > bwaves_1.out 2>> bwaves_1.err"
# command_dict["603_2"]="./speed_bwaves_base.mytest-m64 bwaves_2 < bwaves_2.in > bwaves_2.out 2>> bwaves_2.err"
# command_dict["607"]="./cactuBSSN_s_base.mytest-m64 spec_ref.par 0<&- > spec_ref.out 2>> spec_ref.err"
# command_dict["619"]="./lbm_s_base.mytest-m64 2000 reference.dat 0 0 200_200_260_ldc.of 0<&- > lbm.out 2>> lbm.err"
# command_dict["621"]="./wrf_s_base.mytest-m64 0<&- > rsl.out.0000 2>> wrf.err"
# command_dict["627"]="./cam4_s_base.mytest-m64 0<&- > cam4_s_base.icc_speed_no_affinity.txt 2>> cam4_s_base.icc_speed_no_affinity.err"
# command_dict["628"]="./speed_pop2_base.mytest-m64 0<&- > pop2_s.out 2>> pop2_s.err"
# command_dict["638"]="./imagick_s_base.mytest-m64 -limit disk 0 refspeed_input.tga -resize 817% -rotate -2.76 -shave 540x375 -alpha remove -auto-level -contrast-stretch 1x1% -colorspace Lab -channel R -equalize +channel -colorspace sRGB -define histogram:unique-colors=false -adaptive-blur 0x5 -despeckle -auto-gamma -adaptive-sharpen 55 -enhance -brightness-contrast 10x10 -resize 30% refspeed_output.tga 0<&- > refspeed_convert.out 2>> refspeed_convert.err"
# command_dict["644"]="./nab_s_base.mytest-m64 3j1n 20140317 220 0<&- > 3j1n.out 2>> 3j1n.err"
# command_dict["649"]="./fotonik3d_s_base.mytest-m64 0<&- > fotonik3d_s.log 2>> fotonik3d_s.err"
# command_dict["654"]="./sroms_base.mytest-m64 < ocean_benchmark3.in > ocean_benchmark3.log 2>> ocean_benchmark3.err"



# <intrate refrate>
# command_dict["500_1"]="./perlbench_r_base.mytest-m64 -I./lib checkspam.pl 2500 5 25 11 150 1 1 1 1 0<&- > checkspam.2500.5.25.11.150.1.1.1.1.out 2>> checkspam.2500.5.25.11.150.1.1.1.1.err"
# command_dict["500_2"]="./perlbench_r_base.mytest-m64 -I./lib diffmail.pl 4 800 10 17 19 300 0<&- > diffmail.4.800.10.17.19.300.out 2>> diffmail.4.800.10.17.19.300.err"
# command_dict["500_3"]="./perlbench_r_base.mytest-m64 -I./lib splitmail.pl 6400 12 26 16 100 0 0<&- > splitmail.6400.12.26.16.100.0.out 2>> splitmail.6400.12.26.16.100.0.err"
# command_dict["502_1"]="./cpugcc_r_base.mytest-m64 gcc-pp.c -O3 -finline-limit=0 -fif-conversion -fif-conversion2 -o gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.s 0<&- > gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.out 2>> gcc-pp.opts-O3_-finline-limit_0_-fif-conversion_-fif-conversion2.err"
# command_dict["502_2"]="./cpugcc_r_base.mytest-m64 gcc-pp.c -O2 -finline-limit=36000 -fpic -o gcc-pp.opts-O2_-finline-limit_36000_-fpic.s 0<&- > gcc-pp.opts-O2_-finline-limit_36000_-fpic.out 2>> gcc-pp.opts-O2_-finline-limit_36000_-fpic.err"
# command_dict["502_3"]="./cpugcc_r_base.mytest-m64 gcc-smaller.c -O3 -fipa-pta -o gcc-smaller.opts-O3_-fipa-pta.s 0<&- > gcc-smaller.opts-O3_-fipa-pta.out 2>> gcc-smaller.opts-O3_-fipa-pta.err"
# command_dict["502_4"]="./cpugcc_r_base.mytest-m64 ref32.c -O5 -o ref32.opts-O5.s 0<&- > ref32.opts-O5.out 2>> ref32.opts-O5.err"
# command_dict["502_5"]="./cpugcc_r_base.mytest-m64 ref32.c -O3 -fselective-scheduling -fselective-scheduling2 -o ref32.opts-O3_-fselective-scheduling_-fselective-scheduling2.s 0<&- > ref32.opts-O3_-fselective-scheduling_-fselective-scheduling2.out 2>> ref32.opts-O3_-fselective-scheduling_-fselective-scheduling2.err"
# command_dict["505"]="./mcf_r_base.mytest-m64 inp.in 0<&- > inp.out 2>> inp.err"
# command_dict["520"]="./omnetpp_r_base.mytest-m64 -c General -r 0 0<&- > omnetpp.General-0.out 2>> omnetpp.General-0.err"
# command_dict["523"]="./cpuxalan_r_base.mytest-m64 -v t5.xml xalanc.xsl 0<&- > ref-t5.out 2>> ref-t5.err"
# command_dict["525_1"]="./x264_r_base.mytest-m64 --pass 1 --stats x264_stats.log --bitrate 1000 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720 0<&- > run_000-1000_x264_r_base.icc_rate_no_affinity_x264_pass1.out 2>> run_000-1000_x264_r_base.icc_rate_no_affinity_x264_pass1.err"
# command_dict["525_2"]="./x264_r_base.mytest-m64 --pass 2 --stats x264_stats.log --bitrate 1000 --dumpyuv 200 --frames 1000 -o BuckBunny_New.264 BuckBunny.yuv 1280x720 0<&- > run_000-1000_x264_r_base.icc_rate_no_affinity_x264_pass2.out 2>> run_000-1000_x264_r_base.icc_rate_no_affinity_x264_pass2.err"
# command_dict["525_3"]="./x264_r_base.mytest-m64 --seek 500 --dumpyuv 200 --frames 1250 -o BuckBunny_New.264 BuckBunny.yuv 1280x720 0<&- > run_0500-1250_x264_r_base.icc_rate_no_affinity_x264.out 2>> run_0500-1250_x264_r_base.icc_rate_no_affinity_x264.err"
# command_dict["531"]="./deepsjeng_r_base.mytest-m64 ref.txt 0<&- > ref.out 2>> ref.err"
# command_dict["541"]="./leela_r_base.mytest-m64 ref.sgf 0<&- > ref.out 2>> ref.err"
# command_dict["548"]="./exchange2_r_base.mytest-m64 6 0<&- > exchange2.txt 2>> exchange2.err"
# command_dict["557_1"]="./xz_r_base.mytest-m64 cld.tar.xz 160 19cf30ae51eddcbefda78dd06014b4b96281456e078ca7c13e1c0c9e6aaea8dff3efb4ad6b0456697718cede6bd5454852652806a657bb56e07d61128434b474 59796407 61004416 6 0<&- > cld.tar-160-6.out 2>> cld.tar-160-6.err"
# command_dict["557_2"]="./xz_r_base.mytest-m64 cpu2006docs.tar.xz 250 055ce243071129412e9dd0b3b69a21654033a9b723d874b2015c774fac1553d9713be561ca86f74e4f16f22e664fc17a79f30caa5ad2c04fbc447549c2810fae 23047774 23513385 6e 0<&- > cpu2006docs.tar-250-6e.out 2>> cpu2006docs.tar-250-6e.err"
# command_dict["557_3"]="./xz_r_base.mytest-m64 input.combined.xz 250 a841f68f38572a49d86226b7ff5baeb31bd19dc637a922a972b2e6d1257a890f6a544ecab967c313e370478c74f760eb229d4eef8a8d2836d233d3e9dd1430bf 40401484 41217675 7 0<&- > input.combined-250-7.out 2>> input.combined-250-7.err"



# <fprate refrate>
# command_dict["503_1"]="./bwaves_r_base.mytest-m64 bwaves_1 < bwaves_1.in > bwaves_1.out 2>> bwaves_1.err" 
# command_dict["503_2"]="./bwaves_r_base.mytest-m64 bwaves_2 < bwaves_2.in > bwaves_2.out 2>> bwaves_2.err"
# command_dict["503_3"]="./bwaves_r_base.mytest-m64 bwaves_3 < bwaves_3.in > bwaves_3.out 2>> bwaves_3.err"
# command_dict["503_4"]="./bwaves_r_base.mytest-m64 bwaves_4 < bwaves_4.in > bwaves_4.out 2>> bwaves_4.err"
# command_dict["507"]="./cactusBSSN_r_base.mytest-m64 spec_ref.par 0<&- > spec_ref.out 2>> spec_ref.err"
# command_dict["508"]="./namd_r_base.mytest-m64 --input apoa1.input --output apoa1.ref.output --iterations 65 0<&- > namd.out 2>> namd.err"
# command_dict["510"]="./parest_r_base.mytest-m64 ref.prm 0<&- > ref.out 2>> ref.err"
# command_dict["511"]="./povray_r_base.mytest-m64 SPEC-benchmark-ref.ini 0<&- > SPEC-benchmark-ref.stdout 2>> SPEC-benchmark-ref.stderr"
# command_dict["519"]="./lbm_r_base.mytest-m64 3000 reference.dat 0 0 100_100_130_ldc.of 0<&- > lbm.out 2>> lbm.err"
# command_dict["521"]="./wrf_r_base.mytest-m64 0<&- > rsl.out.0000 2>> wrf.err"
# command_dict["526"]="./blender_r_base.mytest-m64 sh3_no_char.blend --render-output sh3_no_char_ --threads 1 -b -F RAWTGA -s 849 -e 849 -a 0<&- > sh3_no_char.849.spec.out 2>> sh3_no_char.849.spec.err"
# command_dict["527"]="./cam4_r_base.mytest-m64 0<&- > cam4_r_base.icc_rate_no_affinity.txt 2>> cam4_r_base.icc_rate_no_affinity.err"
# command_dict["538"]="./imagick_r_base.mytest-m64 -limit disk 0 refrate_input.tga -edge 41 -resample 181% -emboss 31 -colorspace YUV -mean-shift 19x19+15% -resize 30% refrate_output.tga 0<&- > refrate_convert.out 2>> refrate_convert.err"
# command_dict["544"]="./nab_r_base.mytest-m64 1am0 1122214447 122 0<&- > 1am0.out 2>> 1am0.err"
# command_dict["549"]="./fotonik3d_r_base.mytest-m64 0<&- > fotonik3d_r.log 2>> fotonik3d_r.err"
# command_dict["554"]="./roms_r_base.mytest-m64 < ocean_benchmark2.in.x > ocean_benchmark2.log 2>> ocean_benchmark2.err"


for key in "${!command_dict[@]}"; do
    echo "start benchmark ${key}"
    dirname="${key%_*}"
    exe_path="${path}/${dirname}"
    save_path="${s_path}/${key}"
    python extract_data.py --path "${exe_path}" --exe_cmd "${command_dict[$key]}" --path_save "${save_path}" || continue
    echo "done benchmark ${key}"
done





