aux_source_directory(. SRCS_PARSER)

add_library(kcftracker STATIC ${SRCS_PARSER})
target_link_libraries(kcftracker ${OpenCV_LIBS})

install(TARGETS kcftracker DESTINATION ${PROJECT_SOURCE_DIR}/install/lib)
install(DIRECTORY ${KCFTRACKER_INCLUDE_DIR} DESTINATION ${PROJECT_SOURCE_DIR}/install)
