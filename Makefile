CC=gcc
CXX=clang++-17
CXXFLAGS=-std=c++20 -Wall -march=native

all: bin/1brc bin/release-1brc bin/symbols-1brc bin/gen


bin/1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -ggdb -D_GLIBCXX_DEBUG src/main.cc -o $@

bin/release-1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -static-libstdc++ -DNDEBUG -O3 src/main.cc -o $@

bin/symbols-1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -static-libstdc++  -DNDEBUG -gdwarf-5 -O3 src/main.cc -o $@

bin/stripped-1brc: bin/symbols-1brc
	mkdir -p bin/
	cp $< $@
	objcopy --only-keep-debug $@ bin/stripped-1brc.dbg
	objcopy --strip-debug $@
	objcopy --add-gnu-debuglink=bin/stripped-1brc.dbg $@


bin/gen: utils/gen.c
	$(CC) -O2 $< -o $@ -lm

dump.s: bin/symbols-1brc
	objdump -dCSl -M Intel --no-show-raw-insn $< > $@

data/100M.txt: bin/gen
	./bin/gen 100000000
	mv measurements.txt ./data/100M.txt

perf_record: bin/stripped-1brc bin/symbols-1brc
	sudo perf record -e cpu-cycles:PH,cpu_core/L1-dcache-load-misses/ -F 4096 --call-graph dwarf bin/stripped-1brc data/100M.txt 1 > 100M-test.txt
	sudo perf buildid-cache -u bin/symbols-1brc

bolt: bin/symbols-1brc
	sudo perf record -e cycles:uP -j any,u -o perf_bolt.data -- $< data/100M.txt 1 > 100M-test.txt
	sudo perf2bolt -p perf_bolt.data -o perf_bolt.fdata $<
	sudo llvm-bolt $< -o bin/symbols-1brc-bolt -data perf_bolt.fdata -reorder-blocks=ext-tsp -reorder-functions=hfsort -split-functions -split-all-cold -split-eh -dyno-stats

bin/pgo-1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -DNDEBUG -static-libstdc++ -fprofile-instr-generate -O3 src/main.cc -o $@

bin/pgo-opt-1brc: bin/pgo-1brc
	bin/pgo-1brc data/100M.txt 1 > 100M-test.txt
	llvm-profdata merge -output=pgo-1brc.profdata default.profraw
	$(CXX) $(CXXFLAGS) -DNDEBUG -fprofile-instr-use=pgo-1brc.profdata -O3 src/main.cc -o $@

clean:
	rm -f bin/*
	rm -f dump.s
