CXX=clang++-16
CXXFLAGS=-std=c++20 -Wall -march=native

all: bin/1brc bin/release-1brc bin/symbols-1brc


bin/1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -ggdb src/main.cc -o $@

bin/release-1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -DNDEBUG -O3 src/main.cc -o $@
bin/symbols-1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -DNDEBUG -gdwarf-4 -O3 src/main.cc -o $@

dump.s: bin/symbols-1brc
	objdump -dCSl bin/release-1brc -M Intel > $@

clean:
	rm bin/*
