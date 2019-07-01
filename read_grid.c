#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

int index_to_model(int SPEC_LEN, int T, int CLOUD, int CO, int HAZE , int GRAVITY, int LOGZ){
  return SPEC_LEN * (T + 22*GRAVITY + 88*LOGZ + 440*CO + 1760*HAZE + 7040*CLOUD);
}


int main()
{
  int fd;
  struct stat sb;
  size_t length; // Total length of grid to allocate space for.
  int SPEC_LEN = 5006; // Number of points in one spectrum, contstant.

  // Open file and get file descriptor
  fd = open("../ATMO/ATMO_GRID_REDUCED_FINAL.txt", O_RDONLY);
  // Get size of file and check for error
  if (fstat(fd, &sb) == -1) {exit(EXIT_FAILURE);}
  length = sb.st_size;

  // Map file into random access memory
  double *addr = mmap(NULL, length, PROT_READ, MAP_PRIVATE, fd, 0);

  if (addr == MAP_FAILED) {exit(EXIT_FAILURE);}

  // Write file contents to terminal

  int T = 21; // The nth temperature point
  int CLOUD = 3; // The mth cloud point...
  int CO = 3;
  int HAZE = 3;
  int GRAVITY = 3;
  int LOGZ = 4;

  int model_idx = index_to_model(SPEC_LEN, T, CLOUD, CO, HAZE, GRAVITY, LOGZ);

  for (int i = model_idx; i <= model_idx+SPEC_LEN-1; ++i) {
    printf("%d %.12lf\n", i, addr[i]);
  }
  // Remove mapping, close file. Fin.
  munmap(addr, length);
  close(fd);

  return 1;
}
