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
	$(CXX) $(CXXFLAGS) -static-libstdc++ -DNDEBUG -gdwarf-5 -O3 src/main.cc -o $@

bin/gen: utils/gen.c
	$(CC) -O2 $< -o $@ -lm

dump.s: bin/symbols-1brc
	objdump -dCSl -M Intel --no-show-raw-insn $< > $@

data/100M.txt: bin/gen
	./bin/gen 100000000
	mv measurements.txt ./data/100M.txt

perf_record:
	sudo perf record -e  cpu-cycles:PH -F 8192 --call-graph dwarf -g ./bin/symbols-1brc data/100M.txt 1 > 100M-test.txt

clean:
	rm -f bin/*
	rm -f dump.s
