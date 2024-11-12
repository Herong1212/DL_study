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
CMAKE_SOURCE_DIR = /home/dsg/DL_study/Linear_regression/example1

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dsg/DL_study/Linear_regression/example1/build

# Include any dependencies generated for this target.
include CMakeFiles/singleX.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/singleX.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/singleX.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/singleX.dir/flags.make

CMakeFiles/singleX.dir/singleX.cpp.o: CMakeFiles/singleX.dir/flags.make
CMakeFiles/singleX.dir/singleX.cpp.o: /home/dsg/DL_study/Linear_regression/example1/singleX.cpp
CMakeFiles/singleX.dir/singleX.cpp.o: CMakeFiles/singleX.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dsg/DL_study/Linear_regression/example1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/singleX.dir/singleX.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/singleX.dir/singleX.cpp.o -MF CMakeFiles/singleX.dir/singleX.cpp.o.d -o CMakeFiles/singleX.dir/singleX.cpp.o -c /home/dsg/DL_study/Linear_regression/example1/singleX.cpp

CMakeFiles/singleX.dir/singleX.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/singleX.dir/singleX.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dsg/DL_study/Linear_regression/example1/singleX.cpp > CMakeFiles/singleX.dir/singleX.cpp.i

CMakeFiles/singleX.dir/singleX.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/singleX.dir/singleX.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dsg/DL_study/Linear_regression/example1/singleX.cpp -o CMakeFiles/singleX.dir/singleX.cpp.s

# Object files for target singleX
singleX_OBJECTS = \
"CMakeFiles/singleX.dir/singleX.cpp.o"

# External object files for target singleX
singleX_EXTERNAL_OBJECTS =

singleX: CMakeFiles/singleX.dir/singleX.cpp.o
singleX: CMakeFiles/singleX.dir/build.make
singleX: /usr/local/lib/libtorch/lib/libtorch.so
singleX: /usr/local/lib/libtorch/lib/libc10.so
singleX: /usr/local/lib/libtorch/lib/libkineto.a
singleX: /usr/local/cuda-11.8/lib64/libnvrtc.so
singleX: /usr/local/lib/libtorch/lib/libc10_cuda.so
singleX: /usr/local/lib/libtorch/lib/libc10_cuda.so
singleX: /usr/local/lib/libtorch/lib/libc10.so
singleX: /usr/local/cuda-11.8/lib64/libcudart.so
singleX: /usr/local/cuda-11.8/lib64/libnvToolsExt.so
singleX: CMakeFiles/singleX.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dsg/DL_study/Linear_regression/example1/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Linking CXX executable singleX"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/singleX.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/singleX.dir/build: singleX
.PHONY : CMakeFiles/singleX.dir/build

CMakeFiles/singleX.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/singleX.dir/cmake_clean.cmake
.PHONY : CMakeFiles/singleX.dir/clean

CMakeFiles/singleX.dir/depend:
	cd /home/dsg/DL_study/Linear_regression/example1/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dsg/DL_study/Linear_regression/example1 /home/dsg/DL_study/Linear_regression/example1 /home/dsg/DL_study/Linear_regression/example1/build /home/dsg/DL_study/Linear_regression/example1/build /home/dsg/DL_study/Linear_regression/example1/build/CMakeFiles/singleX.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/singleX.dir/depend
