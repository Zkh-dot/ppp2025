add_executable(invert_pgm invert_pgm.c)
target_link_libraries(invert_pgm
  ppp_pnm
  )

add_executable(invert_pgm_mpi invert_pgm_mpi.c)
target_link_libraries(invert_pgm_mpi
  ppp_pnm
  ${MPI_C_LIBRARIES}
  )
