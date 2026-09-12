# generated from ament/cmake/core/templates/nameConfig.cmake.in

# prevent multiple inclusion
if(_chopped_urdf_v2_CONFIG_INCLUDED)
  # ensure to keep the found flag the same
  if(NOT DEFINED chopped_urdf_v2_FOUND)
    # explicitly set it to FALSE, otherwise CMake will set it to TRUE
    set(chopped_urdf_v2_FOUND FALSE)
  elseif(NOT chopped_urdf_v2_FOUND)
    # use separate condition to avoid uninitialized variable warning
    set(chopped_urdf_v2_FOUND FALSE)
  endif()
  return()
endif()
set(_chopped_urdf_v2_CONFIG_INCLUDED TRUE)

# output package information
if(NOT chopped_urdf_v2_FIND_QUIETLY)
  message(STATUS "Found chopped_urdf_v2: 0.3.0 (${chopped_urdf_v2_DIR})")
endif()

# warn when using a deprecated package
if(NOT "" STREQUAL "")
  set(_msg "Package 'chopped_urdf_v2' is deprecated")
  # append custom deprecation text if available
  if(NOT "" STREQUAL "TRUE")
    set(_msg "${_msg} ()")
  endif()
  # optionally quiet the deprecation message
  if(NOT chopped_urdf_v2_DEPRECATED_QUIET)
    message(DEPRECATION "${_msg}")
  endif()
endif()

# flag package as ament-based to distinguish it after being find_package()-ed
set(chopped_urdf_v2_FOUND_AMENT_PACKAGE TRUE)

# include all config extra files
set(_extras "")
foreach(_extra ${_extras})
  include("${chopped_urdf_v2_DIR}/${_extra}")
endforeach()
