aux_source_directory(. SRCS_TOOLS)

find_package(gflags REQUIRED)
include_directories( ${gflags_INCLUDE_DIRS} )

foreach(source ${SRCS_TOOLS})
  get_filename_component(name ${source} NAME_WE)
  set(name ${name}.bin)

  add_executable(${name} ${source})
  target_link_libraries(${name} kcftracker ${OpenCV_LIBS} gflags)

  install(TARGETS ${name} DESTINATION ${PROJECT_SOURCE_DIR}/install/samples)
endforeach(source)
