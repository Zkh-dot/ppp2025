add_executable(bcast bcast.c)
target_link_libraries(bcast
  ${MPI_C_LIBRARIES}
  m
  )
