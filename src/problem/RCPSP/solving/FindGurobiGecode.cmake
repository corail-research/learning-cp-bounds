find_path(GUROBI_INCLUDE_DIRS
    NAMES gurobi_c.h
    HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
    PATH_SUFFIXES include)

find_library(GUROBI_LIBRARY
    NAMES gurobi gurobi100 gurobi110
    HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
    PATH_SUFFIXES lib)

if(CXX)
    if(MSVC)
        set(MSVC_YEAR "2017")
        
        if(MT)
            set(M_FLAG "mt")
        else()
            set(M_FLAG "md")
        endif()
        
        find_library(GUROBI_CXX_LIBRARY
            NAMES gurobi_c++${M_FLAG}${MSVC_YEAR}
            HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
            PATH_SUFFIXES lib)
        find_library(GUROBI_CXX_DEBUG_LIBRARY
            NAMES gurobi_c++${M_FLAG}d${MSVC_YEAR}
            HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
            PATH_SUFFIXES lib)
    else()
        find_library(GUROBI_CXX_LIBRARY
            NAMES gurobi_c++
            HINTS ${GUROBI_DIR} $ENV{GUROBI_HOME}
            PATH_SUFFIXES lib)
        set(GUROBI_CXX_DEBUG_LIBRARY ${GUROBI_CXX_LIBRARY})
    endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GUROBI DEFAULT_MSG GUROBI_LIBRARY)

set(GECODE_ROOT "/home/bessa75/scratch/gecode-release-6.2.0/")
# - Try to find Gecode
# Once done this will define
#  GECODE_FOUND          - System has Gecode
#  GECODE_INCLUDE_DIRS   - The Gecode include directories
#  GECODE_LIBRARIES      - The libraries needed to use Gecode
#  GECODE_TARGETS        - The names of imported targets created for gecode
# User can set GECODE_ROOT to the preferred installation prefix

list(INSERT CMAKE_PREFIX_PATH 0 "${GECODE_ROOT}" "$ENV{GECODE_ROOT}")

find_path(GECODE_INCLUDE gecode/kernel.hh
          PATH_SUFFIXES include)

find_file(GECODE_CONFIG_LOC gecode/support/config.hpp
          HINTS ${GECODE_INCLUDE}
          PATH_SUFFIXES include)

if(NOT "${GECODE_CONFIG_LOC}" STREQUAL "GECODE_CONFIG_LOC-NOTFOUND")
  file(READ "${GECODE_CONFIG_LOC}" GECODE_CONFIG)
  string(REGEX MATCH "\#define GECODE_VERSION \"([0-9]+.[0-9]+.[0-9]+)\"" _ "${GECODE_CONFIG}")
  set(GECODE_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "\#define GECODE_LIBRARY_VERSION \"([0-9]+-[0-9]+-[0-9]+)\"" _ "${GECODE_CONFIG}")
  set(GECODE_LIBRARY_VERSION "${CMAKE_MATCH_1}")
  string(REGEX MATCH "\#define GECODE_STATIC_LIBS ([0-9]+)" _ "${GECODE_CONFIG}")
  set(GECODE_STATIC_LIBS "${CMAKE_MATCH_1}")
  string(REGEX MATCH "\#define GECODE_HAS_GIST" GECODE_HAS_GIST "${GECODE_CONFIG}")
endif()

set(GECODE_COMPONENTS Driver Flatzinc Float Int Kernel Minimodel Search Set Support)
if(GECODE_HAS_GIST)
  list(APPEND GECODE_COMPONENTS Gist)
endif()

foreach(GECODE_COMP ${GECODE_COMPONENTS})
  # Try to find gecode library
  string(TOLOWER "gecode${GECODE_COMP}" GECODE_LIB)
  set(GECODE_LIB_LOC "GECODE_LIB_LOC-NOTFOUND")
  find_library(GECODE_LIB_LOC NAMES ${GECODE_LIB} lib${GECODE_LIB} ${GECODE_LIB}-${GECODE_LIBRARY_VERSION}-r-x64 ${GECODE_LIB}-${GECODE_LIBRARY_VERSION}-d-x64
               HINTS ${GECODE_INCLUDE}
               PATH_SUFFIXES lib)
  if(NOT "${GECODE_LIB_LOC}" STREQUAL "GECODE_LIB_LOC-NOTFOUND")
      list(APPEND GECODE_LIBRARY ${GECODE_LIB_LOC})
      add_library(Gecode::${GECODE_COMP} UNKNOWN IMPORTED)
      set_target_properties(Gecode::${GECODE_COMP} PROPERTIES
                            IMPORTED_LOCATION ${GECODE_LIB_LOC}
                            INTERFACE_INCLUDE_DIRECTORIES ${GECODE_INCLUDE})
      set(Gecode_FIND_REQUIRED_${GECODE_COMP} TRUE)
      set(Gecode_${GECODE_COMP}_FOUND TRUE)
  endif()
endforeach(GECODE_COMP)

if(WIN32 AND GECODE_HAS_GIST AND GECODE_STATIC_LIBS)
  find_package(Qt5 QUIET COMPONENTS Core Gui Widgets PrintSupport)
  set_target_properties(Gecode::Gist PROPERTIES
                        INTERFACE_LINK_LIBRARIES "Qt5::Core;Qt5::Gui;Qt5::Widgets;Qt5::PrintSupport")
endif()

unset(GECODE_REQ_LIBS)
unset(GECODE_LIB_WIN)
unset(GECODE_LIB_LOC)



include(FindPackageHandleStandardArgs)
# handle the QUIETLY and REQUIRED arguments and set GECODE_FOUND to TRUE
# if all listed variables are TRUE
find_package_handle_standard_args(
  Gecode
  REQUIRED_VARS GECODE_INCLUDE GECODE_LIBRARY
  VERSION_VAR GECODE_VERSION
  HANDLE_COMPONENTS
)

mark_as_advanced(GECODE_INCLUDE GECODE_LIBRARY)
list(REMOVE_AT CMAKE_PREFIX_PATH 1 0)

set(GECODE_LIBRARIES ${GECODE_LIBRARY})
set(GECODE_INCLUDE_DIRS ${GECODE_INCLUDE})
