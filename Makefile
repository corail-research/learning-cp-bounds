PYTHON_PATH="/home/bourgeat/anaconda3/bin/python3"

binpacking:
	rm -rf src/problem/binpacking/solving/build
	cmake -Hsrc/problem/binpacking/solving -Bsrc/problem/binpacking/solving/build -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/binpacking/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/binpacking/solving/build/solver_binpacking ./

mknapsack:
	rm -rf src/problem/mknapsack/solving/build
	cmake -Hsrc/problem/mknapsack/solving -Bsrc/problem/mknapsack/solving/build -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH) -DCMAKE_EXPORT_COMPILE_COMMANDS:BOOL=TRUE -DCMAKE_BUILD_TYPE:STRING="Release" -G "Unix Makefiles" -DPYTHON_EXECUTABLE:FILEPATH=$(PYTHON_PATH)
	cmake --build src/problem/mknapsack/solving/build --config Release --target all -- -j $(nproc) VERBOSE=1
	mv src/problem/mknapsack/solving/build/solver_mknaps ./
