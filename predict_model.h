/* soure types */
#define STYPE_POINT 0
#define STYPE_GAUSSIAN 1
#define STYPE_DISK 2
#define STYPE_RING 3
#define STYPE_SHAPELET 4

#ifndef CONST_C
#define CONST_C 299792458.0
#endif
#ifndef DEFAULT_TH_PER_BK
#define DEFAULT_TH_PER_BK 64
#endif

typedef struct exinfo_gaussian_ {
  double eX,eY,eP; /* major,minor,PA */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_gaussian;

typedef struct exinfo_disk_ {
  double eX; /* diameter */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_disk;

typedef struct exinfo_ring_ {
  double eX; /* diameter */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_ring;

typedef struct exinfo_shapelet_ {
  int n0; /* model order, no of modes=n0*n0 */
  double beta; /* scale */
  double *modes; /* array of n0*n0 x 1 values */
  double eX,eY,eP; /* linear transform parameters */

  double cxi,sxi,cphi,sphi; /* projection of [0,0,1] to [l,m,n] */
  int use_projection;
} exinfo_shapelet;

typedef struct baseline_t_ {
 int sta1,sta2;
 unsigned char flag; /* if this baseline is flagged, set to 1, otherwise 0: 
             special case: 2 if baseline is not used in solution, but will be
              subtracted */
} baseline_t;
