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
CMAKE_SOURCE_DIR = /home/dsg/DL_study/NeuralNet/purchase_product

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/dsg/DL_study/NeuralNet/purchase_product/build

# Include any dependencies generated for this target.
include CMakeFiles/purchase_product.dir/depend.make
# Include any dependencies generated by the compiler for this target.
include CMakeFiles/purchase_product.dir/compiler_depend.make

# Include the progress variables for this target.
include CMakeFiles/purchase_product.dir/progress.make

# Include the compile flags for this target's objects.
include CMakeFiles/purchase_product.dir/flags.make

CMakeFiles/purchase_product.dir/purchase_product.cpp.o: CMakeFiles/purchase_product.dir/flags.make
CMakeFiles/purchase_product.dir/purchase_product.cpp.o: /home/dsg/DL_study/NeuralNet/purchase_product/purchase_product.cpp
CMakeFiles/purchase_product.dir/purchase_product.cpp.o: CMakeFiles/purchase_product.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dsg/DL_study/NeuralNet/purchase_product/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_1) "Building CXX object CMakeFiles/purchase_product.dir/purchase_product.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/purchase_product.dir/purchase_product.cpp.o -MF CMakeFiles/purchase_product.dir/purchase_product.cpp.o.d -o CMakeFiles/purchase_product.dir/purchase_product.cpp.o -c /home/dsg/DL_study/NeuralNet/purchase_product/purchase_product.cpp

CMakeFiles/purchase_product.dir/purchase_product.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/purchase_product.dir/purchase_product.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dsg/DL_study/NeuralNet/purchase_product/purchase_product.cpp > CMakeFiles/purchase_product.dir/purchase_product.cpp.i

CMakeFiles/purchase_product.dir/purchase_product.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/purchase_product.dir/purchase_product.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dsg/DL_study/NeuralNet/purchase_product/purchase_product.cpp -o CMakeFiles/purchase_product.dir/purchase_product.cpp.s

CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o: CMakeFiles/purchase_product.dir/flags.make
CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o: /home/dsg/DL_study/NeuralNet/purchase_product/src/Purchase_product.cpp
CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o: CMakeFiles/purchase_product.dir/compiler_depend.ts
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --progress-dir=/home/dsg/DL_study/NeuralNet/purchase_product/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_2) "Building CXX object CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -MD -MT CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o -MF CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o.d -o CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o -c /home/dsg/DL_study/NeuralNet/purchase_product/src/Purchase_product.cpp

CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.i"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -E /home/dsg/DL_study/NeuralNet/purchase_product/src/Purchase_product.cpp > CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.i

CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.s"
	/usr/bin/c++ $(CXX_DEFINES) $(CXX_INCLUDES) $(CXX_FLAGS) -S /home/dsg/DL_study/NeuralNet/purchase_product/src/Purchase_product.cpp -o CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.s

# Object files for target purchase_product
purchase_product_OBJECTS = \
"CMakeFiles/purchase_product.dir/purchase_product.cpp.o" \
"CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o"

# External object files for target purchase_product
purchase_product_EXTERNAL_OBJECTS =

purchase_product: CMakeFiles/purchase_product.dir/purchase_product.cpp.o
purchase_product: CMakeFiles/purchase_product.dir/src/Purchase_product.cpp.o
purchase_product: CMakeFiles/purchase_product.dir/build.make
purchase_product: CMakeFiles/purchase_product.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green --bold --progress-dir=/home/dsg/DL_study/NeuralNet/purchase_product/build/CMakeFiles --progress-num=$(CMAKE_PROGRESS_3) "Linking CXX executable purchase_product"
	$(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/purchase_product.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
CMakeFiles/purchase_product.dir/build: purchase_product
.PHONY : CMakeFiles/purchase_product.dir/build

CMakeFiles/purchase_product.dir/clean:
	$(CMAKE_COMMAND) -P CMakeFiles/purchase_product.dir/cmake_clean.cmake
.PHONY : CMakeFiles/purchase_product.dir/clean

CMakeFiles/purchase_product.dir/depend:
	cd /home/dsg/DL_study/NeuralNet/purchase_product/build && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/dsg/DL_study/NeuralNet/purchase_product /home/dsg/DL_study/NeuralNet/purchase_product /home/dsg/DL_study/NeuralNet/purchase_product/build /home/dsg/DL_study/NeuralNet/purchase_product/build /home/dsg/DL_study/NeuralNet/purchase_product/build/CMakeFiles/purchase_product.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : CMakeFiles/purchase_product.dir/depend

