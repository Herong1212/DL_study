# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 3.24

# Delete rule output on recipe failure.
.DELETE_ON_ERROR:

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Disable VCS-based implicit rules.
% : %,v

# Disable VCS-based implicit rules.
% : RCS/%

# Disable VCS-based implicit rules.
% : RCS/%,v

# Disable VCS-based implicit rules.
% : SCCS/s.%

# Disable VCS-based implicit rules.
% : s.%

.SUFFIXES: .hpux_make_needs_suffix_list

# Command-line flag to silence nested $(MAKE).
$(VERBOSE)MAKESILENT = -s

#Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/local/bin/cmake

# The command to remove a file.
RM = /usr/local/bin/cmake -E rm -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/dsg/DL_study/NeuralNet

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dsg/DL_study/NeuralNet/build

# Include any dependencies generated for this target.
include example1/CMakeFiles/main.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include example1/CMakeFiles/main.dir/compiler_depend.make

# Include the progress variables for this target.
include example1/CMakeFiles/main.dir/progress.make

# Include the compile flags for this target's objects.
include example1/CMakeFiles/main.dir/flags.make

example1/CMakeFiles/main.dir/main.cpp.o: example1/CMakeFiles/main.dir/flags.make
example1/CMakeFiles/main.dir/main.cpp.o: /home/dsg/DL_study/NeuralNet/example1/main.cpp
example1/CMakeFiles/main.dir/main.cpp.o: example1/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dsg/DL_study/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object example1/CMakeFiles/main.dir/main.cpp.o"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT example1/CMakeFiles/main.dir/main.cpp.o -MF CMakeFiles/main.dir/main.cpp.o.d -o CMakeFiles/main.dir/main.cpp.o -c /home/dsg/DL_study/NeuralNet/example1/main.cpp

example1/CMakeFiles/main.dir/main.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/main.cpp.i"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dsg/DL_study/NeuralNet/example1/main.cpp > CMakeFiles/main.dir/main.cpp.i

example1/CMakeFiles/main.dir/main.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/main.cpp.s"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dsg/DL_study/NeuralNet/example1/main.cpp -o CMakeFiles/main.dir/main.cpp.s

example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.o: example1/CMakeFiles/main.dir/flags.make
example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.o: /home/dsg/DL_study/NeuralNet/example1/src/NeuralNetwork.cpp
example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.o: example1/CMakeFiles/main.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dsg/DL_study/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.o"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.o -MF CMakeFiles/main.dir/src/NeuralNetwork.cpp.o.d -o CMakeFiles/main.dir/src/NeuralNetwork.cpp.o -c /home/dsg/DL_study/NeuralNet/example1/src/NeuralNetwork.cpp

example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/main.dir/src/NeuralNetwork.cpp.i"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dsg/DL_study/NeuralNet/example1/src/NeuralNetwork.cpp > CMakeFiles/main.dir/src/NeuralNetwork.cpp.i

example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/main.dir/src/NeuralNetwork.cpp.s"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && /usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dsg/DL_study/NeuralNet/example1/src/NeuralNetwork.cpp -o CMakeFiles/main.dir/src/NeuralNetwork.cpp.s

# Object files for target main
main_OBJECTS = \
"CMakeFiles/main.dir/main.cpp.o" \
"CMakeFiles/main.dir/src/NeuralNetwork.cpp.o"

# External object files for target main
main_EXTERNAL_OBJECTS =

example1/main: example1/CMakeFiles/main.dir/main.cpp.o
example1/main: example1/CMakeFiles/main.dir/src/NeuralNetwork.cpp.o
example1/main: example1/CMakeFiles/main.dir/build.make
example1/main: /usr/local/lib/libtorch/lib/libtorch.so
example1/main: /usr/local/lib/libtorch/lib/libc10.so
example1/main: /usr/local/lib/libtorch/lib/libkineto.a
example1/main: /usr/local/cuda-11.8/lib64/libnvrtc.so
example1/main: /usr/local/lib/libtorch/lib/libc10_cuda.so
example1/main: /usr/local/lib/libtorch/lib/libc10_cuda.so
example1/main: /usr/local/lib/libtorch/lib/libc10.so
example1/main: /usr/local/cuda-11.8/lib64/libcudart.so
example1/main: /usr/local/cuda-11.8/lib64/libnvToolsExt.so
example1/main: example1/CMakeFiles/main.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dsg/DL_study/NeuralNet/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable main"
	cd /home/dsg/DL_study/NeuralNet/build/example1 && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/main.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
example1/CMakeFiles/main.dir/build: example1/main
.PHONY : example1/CMakeFiles/main.dir/build

example1/CMakeFiles/main.dir/clean:
	cd /home/dsg/DL_study/NeuralNet/build/example1 && $(CMAKE_COMMAND) -P CMakeFiles/main.dir/cmake_clean.cmake
.PHONY : example1/CMakeFiles/main.dir/clean

example1/CMakeFiles/main.dir/depend:
	cd /home/dsg/DL_study/NeuralNet/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dsg/DL_study/NeuralNet /home/dsg/DL_study/NeuralNet/example1 /home/dsg/DL_study/NeuralNet/build /home/dsg/DL_study/NeuralNet/build/example1 /home/dsg/DL_study/NeuralNet/build/example1/CMakeFiles/main.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : example1/CMakeFiles/main.dir/depend
