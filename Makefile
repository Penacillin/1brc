CXX=clang++-16
CXXFLAGS=-std=c++20 -Wall -march=native

all: bin/1brc bin/release-1brc


bin/1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -ggdb src/main.cc -o $@

bin/release-1brc: $(wildcard src/*)
	mkdir -p bin/
	$(CXX) $(CXXFLAGS) -DNDEBUG -O3 src/main.cc -o $@


clean:
	rm bin/*
