/*
 * This file is a part of TiledArray.
 * Copyright (C) 2013  Virginia Tech
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 */

#include <TiledArray/cuda/btas_um_tensor.h>
#include <TiledArray/version.h>
#include <tiledarray.h>
#include <iostream>

bool to_bool(const char* str) {
  if (not strcmp(str, "0") || not strcmp(str, "no") || not strcmp(str, "false"))
    return false;
  if (not strcmp(str, "1") || not strcmp(str, "yes") || not strcmp(str, "true"))
    return true;
  throw std::runtime_error("unrecognized string specification of bool");
}

// makes tiles of fluctuating sizes
// if n = average tile size
// this will produce tiles of these sizes: n+1, n-1, n+2, n-2, etc.
// the last tile absorbs the remainder
std::vector<unsigned int> make_tiling(unsigned int range_size,
                                      unsigned int ntiles) {
  const auto average_tile_size = range_size / ntiles;
  TA_ASSERT(average_tile_size > ntiles);
  std::vector<unsigned int> result(ntiles + 1);
  result[0] = 0;
  for (long t = 0; t != ntiles - 1; ++t) {
    result[t + 1] =
        result[t] + average_tile_size + ((t % 2 == 0) ? (t + 1) : (-t));
  }
  result[ntiles] = range_size;
  return result;
}

template <typename Tile, typename Policy>
void rand_fill_array(TA::DistArray<Tile, Policy>& array);

template <typename T>
void cs_ccsd(madness::World& world, const TA::TiledRange1& noa,
             const TA::TiledRange1& nva, const TA::TiledRange1& nvc, long repeat);

int main(int argc, char** argv) {
  int rc = 0;

  try {
    // Initialize runtime
    TA::World& world = TA_SCOPED_INITIALIZE(argc, argv);

    // Get command line arguments
    if (argc < 7) {
      std::cout << "Mocks TAMM CD-CCSD (closed-shell)"
                << std::endl
                << "Usage: " << argv[0]
                << " occ_size occ_nblocks uocc_size "
                   "uocc_nblocks n_chol_vecs nblk_chol [repetitions]"
                << std::endl;
      return 0;
    }
    const long n_occ = atol(argv[1]);
    const long nblk_occ = atol(argv[2]);
    const long n_uocc = atol(argv[3]);
    const long nblk_uocc = atol(argv[4]);
    const long n_chol = atol(argv[5]);
    const long nblk_chol = atol(argv[6]);
    if (n_occ <= 0) {
      std::cerr << "Error: occ_size must be greater than zero.\n";
      return 1;
    }
    if (nblk_occ <= 0) {
      std::cerr << "Error: occ_nblocks must be greater than zero.\n";
      return 1;
    }
    if (n_uocc <= 0) {
      std::cerr << "Error: uocc_size must be greater than zero.\n";
      return 1;
    }
    if (nblk_uocc <= 0) {
      std::cerr << "Error: uocc_nblocks must be greater than zero.\n";
      return 1;
    }
    if ((n_occ < nblk_occ) != 0ul) {
      std::cerr << "Error: occ_size must be greater than occ_nblocks.\n";
      return 1;
    }
    if ((n_uocc < nblk_uocc) != 0ul) {
      std::cerr << "Error: uocc_size must be greater than uocc_nblocks.\n";
      return 1;
    }
    const long repeat = (argc >= 8 ? atol(argv[7]) : 5);
    if (repeat <= 0) {
      std::cerr << "Error: number of repetitions must be greater than zero.\n";
      return 1;
    }

    if (world.rank() == 0) {
      std::cout << "TiledArray: TAMM CD-CCSD (closed-shell) test..."
                << "\nGit description: " << TiledArray::git_description()
                << "\nNumber of nodes     = " << world.size()
                << "\nocc size            = " << n_occ
                << "\nocc nblocks         = " << nblk_occ
                << "\nuocc size           = " << n_uocc
                << "\nuocc nblocks        = " << nblk_uocc
                << "\ncvecs size          = " << n_chol
                << "\ncvecs nblocks       = " << nblk_chol
                << "\niterations          = " << repeat                
                << "\nprecision           = " << "double\n";
    }

    // Construct TiledRange1's
    std::vector<unsigned int> tiling_occ = make_tiling(n_occ, nblk_occ);
    std::vector<unsigned int> tiling_uocc = make_tiling(n_uocc, nblk_uocc);
    std::vector<unsigned int> tiling_cv = make_tiling(n_chol, nblk_chol);
    auto noa = TA::TiledRange1(tiling_occ.begin(), tiling_occ.end());
    auto nva = TA::TiledRange1(tiling_uocc.begin(), tiling_uocc.end());
    auto ncv = TA::TiledRange1(tiling_cv.begin(), tiling_cv.end());

    cs_ccsd<double>(world, noa, nva, ncv, repeat);

  } catch (TA::Exception& e) {
    std::cerr << "!! TiledArray exception: " << e.what() << "\n";
    rc = 1;
  } catch (madness::MadnessException& e) {
    std::cerr << "!! MADNESS exception: " << e.what() << "\n";
    rc = 1;
  } catch (SafeMPI::Exception& e) {
    std::cerr << "!! SafeMPI exception: " << e.what() << "\n";
    rc = 1;
  } catch (std::exception& e) {
    std::cerr << "!! std exception: " << e.what() << "\n";
    rc = 1;
  } catch (...) {
    std::cerr << "!! exception: unknown exception\n";
    rc = 1;
  }

  return rc;
}

template <typename T>
void cs_ccsd(TA::World& world, const TA::TiledRange1& noa,
             const TA::TiledRange1& nva, const TA::TiledRange1& ncvecs, long repeat) {

  using CUDATile =
      btas::Tensor<T, TA::Range, TiledArray::cuda_um_btas_varray<T>>;
  using CUDAMatrix = TA::DistArray<TA::Tile<CUDATile>>;

  TA::TiledRange1 nob=noa;
  TA::TiledRange1 nvb=nva;

  TA::TiledRange ncv({ncvecs});
  TA::TiledRange OaOa({noa,noa});
  TA::TiledRange VaOa({nva,noa});
  TA::TiledRange VaVa({nva,nva});
  TA::TiledRange OaVa({noa,nva});
  TA::TiledRange VbVb({nvb,nvb});
  TA::TiledRange ObOb({nob,nob});

  TA::TiledRange OaVaCv({noa,nva,ncvecs});
  TA::TiledRange OaOaCv({noa,noa,ncvecs});
  TA::TiledRange VaOaCv({nva,noa,ncvecs});
  TA::TiledRange VaVaCv({nva,nva,ncvecs});
  TA::TiledRange ObObCv({nob,nob,ncvecs});
  TA::TiledRange VbVbCv({nvb,nvb,ncvecs});
  TA::TiledRange VbObCv({nvb,nob,ncvecs});

  TA::TiledRange VaVaOaOa({nva,nva,noa,noa});
  TA::TiledRange VaVbOaOb({nva,nvb,noa,nob});
  TA::TiledRange OaObOaOb({noa,nob,noa,nob});
  TA::TiledRange VaOaVaOa({nva,noa,nva,noa});
  TA::TiledRange VbVaObOa({nvb,nva,nob,noa});
  TA::TiledRange VbOaVbOa({nvb,noa,nvb,noa});
  TA::TiledRange VbOaVaOb({nvb,noa,nva,nob});
  TA::TiledRange VbObVbOb({nvb,nob,nvb,nob});
  TA::TiledRange VaVbVaVb({nva,nvb,nva,nvb});

  world.gop.fence();
  const double init_t1 = madness::wall_time();

  // Construct tensors
  CUDAMatrix t1_aa(world, VaOa);
  CUDAMatrix f1_oo_aa(world, OaOa);
  CUDAMatrix f1_ov_aa(world, OaVa);
  CUDAMatrix f1_vv_aa(world, VaVa);

  CUDAMatrix t2_aaaa(world,VaVaOaOa);
  CUDAMatrix t2_abab(world,VaVbOaOb);
  CUDAMatrix t2_aaaa_temp(world,VaVaOaOa);

  CUDAMatrix i0_aa(world,VaOa);
  CUDAMatrix i0_abab(world,VaVbOaOb);
  CUDAMatrix r2_abab(world,VaVbOaOb);

  CUDAMatrix chol3d_oo_aa(world,OaOaCv);
  CUDAMatrix chol3d_ov_aa(world,OaVaCv);
  CUDAMatrix chol3d_vv_aa(world,VaVaCv);

  // Energy tensors
  double de = 0.0;
  CUDAMatrix _a01V(world,ncv); //Cv
  CUDAMatrix _a02_aa(world,OaOaCv); //OaOaCv
  CUDAMatrix _a03_aa(world,OaVaCv); //OaVaCv

  // T1 tensors
  CUDAMatrix _a02V(world,ncv); //Cv
  CUDAMatrix _a01_aa(world,OaOaCv);
  CUDAMatrix _a04_aa(world,OaOa);
  CUDAMatrix _a05_aa(world,OaVa);
  CUDAMatrix _a06_aa(world,VaOaCv);

  // T2 tensors
  CUDAMatrix _a007V(world,ncv); //Cv
  CUDAMatrix _a001_aa(world,VaVa);
  CUDAMatrix _a006_aa(world,OaOa);
  CUDAMatrix _a008_aa(world,OaOaCv);
  CUDAMatrix _a009_aa(world,OaOaCv);
  CUDAMatrix _a017_aa(world,VaOaCv);
  CUDAMatrix _a021_aa(world,VaVaCv);

  CUDAMatrix _a001_bb(world,VbVb);
  CUDAMatrix _a006_bb(world,ObOb);
  CUDAMatrix _a009_bb(world,ObObCv);
  CUDAMatrix _a017_bb(world,VbObCv);
  CUDAMatrix _a021_bb(world,VbVbCv);  

  CUDAMatrix _a004_aaaa(VaVaOaOa);
  CUDAMatrix _a004_abab(VaVbOaOb);

  CUDAMatrix _a019_abab(world,OaObOaOb);
  CUDAMatrix _a020_aaaa(world,VaOaVaOa);
  CUDAMatrix _a020_baba(world,VbOaVbOa);
  CUDAMatrix _a020_baab(world,VbOaVaOb);
  CUDAMatrix _a020_bbbb(world,VbObVbOb);
  CUDAMatrix _a022_abab(world,VaVbVaVb);

  CUDAMatrix i0_temp(world,VbVaObOa);


  // Initialize
  rand_fill_array(t1_aa);
  rand_fill_array(t1_aa);
  rand_fill_array(f1_oo_aa);
  rand_fill_array(f1_ov_aa);
  rand_fill_array(f1_vv_aa);
  rand_fill_array(t2_aaaa);
  rand_fill_array(t2_abab);
  rand_fill_array(t2_aaaa_temp);
  rand_fill_array(chol3d_oo_aa);
  rand_fill_array(chol3d_ov_aa);
  rand_fill_array(chol3d_vv_aa);
  rand_fill_array(_a004_aaaa);
  rand_fill_array(_a004_abab);
  
  world.gop.fence();

  const double init_t2 = madness::wall_time();
  const double init_time = init_t2 - init_t1;

  world.gop.fence();
  if (world.rank() == 0) {
    std::cout << "TA init time: " << init_time << "\n";
  }

  double total_time = 0.0;
  world.gop.fence();

  for (int i = 0; i < repeat; ++i) {
    const double start = madness::wall_time();

  /***** CCSD energy terms (closed-shell) ******/
  // t2_aaaa_temp("i,j,k,l") = 0.0;
  t2_aaaa("i,j,k,l")       = t2_abab("i,j,k,l"); 
  t2_aaaa_temp("i,j,k,l")  = t2_aaaa("i,j,k,l"); 
  t2_aaaa("i,j,k,l")      += -1.0 * t2_aaaa_temp("i,j,k,l");
  t2_aaaa_temp("i,j,k,l") += t2_aaaa("i,j,k,l");

  _a01V("c")        =        t1_aa("i,j") * chol3d_ov_aa("j,i,c");
  _a02_aa("i,j,c")  =        t1_aa("a,i") * chol3d_ov_aa("j,a,c");
  _a03_aa("i,a,c")  =        t2_aaaa_temp("a,b,i,j") * chol3d_ov_aa("j,b,c");
  // de                =  2.0 * _a01V("c") * _a01V("c");
  // de               += -1.0 * _a02_aa("i,j,c") * _a02_aa("j,i,c");
  // de               += -1.0 * _a03_aa("i,a,c") * chol3d_ov_aa("i,a,c");
  // de               +=  2.0 * t1_aa("a,i") * f1_ov_aa("i,a");  

  // /***** CCSD T1 terms (closed-shell) ******/
  i0_aa("a,i")      =  1.0 * f1_ov_aa("i,a");
  _a01_aa("i,j,c")  =  1.0 * t1_aa("a,j") * chol3d_ov_aa("i,a,c");
  _a02V("c")        =  2.0 * t1_aa("a,i") * chol3d_ov_aa("i,a,c");
  _a05_aa("i,a")    = -1.0 * chol3d_ov_aa("j,a,c") * _a01_aa("i,j,c");
  _a05_aa("i,a")   +=  1.0 * f1_ov_aa("i,a");

  _a06_aa("a,i,c")  = -1.0 * t2_aaaa_temp("a,b,i,j") * chol3d_ov_aa("j,b,c");
  _a04_aa("j,i")    =  1.0 * f1_oo_aa("j,i");
  _a04_aa("j,i")   +=  1.0 * chol3d_ov_aa("j,a,c") * _a06_aa("a,i,c");
  _a04_aa("j,i")   += -1.0 * t1_aa("a,i") * f1_ov_aa("j,a");
  i0_aa("a,i")     +=  1.0 * t1_aa("a,j") * _a04_aa("j,i");
  i0_aa("a,j")     +=  1.0 * chol3d_ov_aa("j,a,c") * _a02V("c");
  i0_aa("a,j")     +=  1.0 * t2_aaaa_temp("a,b,j,i") * _a05_aa("i,b");
  i0_aa("b,i")     += -1.0 * chol3d_vv_aa("b,a,c") * _a06_aa("a,i,c");
  _a06_aa("b,j,c") += -1.0 * t1_aa("a,j") * chol3d_vv_aa("b,a,c");
  i0_aa("a,j")     += -1.0 * _a06_aa("a,j,c") * _a02V("c");
  _a06_aa("b,i,c") += -1.0 * t1_aa("b,i") * _a02V("c");
  _a06_aa("b,i,c") +=  1.0 * t1_aa("b,j") * _a01_aa("j,i,c");
  _a01_aa("j,i,c") +=  1.0 * chol3d_oo_aa("j,i,c");
  i0_aa("b,i")     +=  1.0 * _a01_aa("j,i,c") * _a06_aa("b,j,c");
  i0_aa("b,i")     +=  1.0 * t1_aa("a,i") * f1_vv_aa("b,a");

  // /***** CCSD T2 terms (closed-shell) ******/
  _a017_aa("a,j,c")   =  -1.0 * t2_aaaa_temp("a,b,j,i") * chol3d_ov_aa("i,b,c");
  _a006_aa("j,i")     =  -1.0 * chol3d_ov_aa("j,b,c") * _a017_aa("b,i,c");
  _a007V("c")         =   2.0 * chol3d_ov_aa("i,a,c") * t1_aa("a,i");
  _a009_aa("i,j,c")   =   1.0 * chol3d_ov_aa("i,a,c") * t1_aa("a,j");
  _a021_aa("b,a,c")   =  -0.5 * chol3d_ov_aa("i,a,c") * t1_aa("b,i");
  _a021_aa("b,a,c")  +=   0.5 * chol3d_vv_aa("b,a,c");
  _a017_aa("a,j,c")  +=  -2.0 * t1_aa("b,j") * _a021_aa("a,b,c");
  _a008_aa("i,j,c")   =   1.0 * _a009_aa("i,j,c");
  _a009_aa("j,i,c")  +=   1.0 * chol3d_oo_aa("i,j,c");
  _a009_bb("j,i,c")   =  _a009_aa("j,i,c");
  _a021_bb("b,a,c")   =  _a021_aa("b,a,c");
  _a001_aa("a,b")     =  -2.0 * _a021_aa("a,b,c") * _a007V("c");
  _a001_aa("a,b")    +=  -1.0 * _a017_aa("a,j,c") * chol3d_ov_aa("j,b,c");
  _a006_aa("j,i")    +=   1.0 * _a009_aa("j,i,c") * _a007V("c");
  _a006_aa("k,i")    +=  -1.0 * _a009_aa("j,i,c") * _a008_aa("k,j,c");

  _a019_abab("j,m,i,n")  =  0.25 * _a009_aa("j,i,c") * _a009_bb("m,n,c");
  _a020_aaaa("b,j,a,i")  = -2.0  * _a009_aa("j,i,c") * _a021_aa("b,a,c");
  _a020_baba("d,j,c,i")  =         _a020_aaaa("d,j,c,i");
  _a020_aaaa("a,k,e,j") +=  0.5  * _a004_aaaa("b,e,k,i") * t2_aaaa("a,b,i,j");
  _a020_baab("c,j,a,n")  = -0.5  * _a004_aaaa("b,a,j,i") * t2_abab("b,c,i,n");
  _a020_baba("c,i,d,j") +=  0.5  * _a004_abab("a,d,i,m") * t2_abab("a,c,j,m");
  _a017_aa("a,j,c")     +=  1.0  * t1_aa("a,i") * chol3d_oo_aa("i,j,c");
  _a017_aa("a,j,c")     += -1.0  * chol3d_ov_aa("j,a,c");
  _a001_aa("b,a")       += -1.0  * f1_vv_aa("b,a");
  _a001_aa("b,a")       +=  1.0  * t1_aa("b,i") * f1_ov_aa("i,a");
  _a006_aa("j,i")       +=  1.0  * f1_oo_aa("j,i");
  _a006_aa("j,i")       +=  1.0  * t1_aa("a,i") * f1_ov_aa("j,a");

  _a017_bb("c,m,v")      = _a017_aa("c,m,v");
  _a006_bb("m,n")        = _a006_aa("m,n");
  _a001_bb("c,d")        = _a001_aa("c,d");
  _a021_bb("c,d,v")      = _a021_aa("c,d,v");
  _a020_bbbb("c,m,d,n")  = _a020_aaaa("c,m,d,n");

  i0_abab("a,d,j,m")     =  1.0 * _a020_bbbb("d,n,c,m") * t2_abab("a,c,j,n");
  i0_abab("b,c,j,m")    +=  1.0 * _a020_baab("c,i,a,m") * t2_aaaa("b,a,j,i");
  i0_abab("a,c,j,m")    +=  1.0 * _a020_baba("c,i,d,j") * t2_abab("a,d,i,m");
  i0_temp("c,a,n,i")     =  i0_abab("c,a,n,i");
  i0_abab("a,c,j,m")    +=  1.0 * i0_temp("c,a,m,j");
  i0_abab("a,c,i,n")    +=  1.0 * _a017_aa("a,i,v") * _a017_bb("c,n,v");
  _a022_abab("a,d,b,c")  =  1.0 * _a021_aa("a,b,v") * _a021_bb("d,c,v");
  i0_abab("a,d,i,n")    +=  4.0 * _a022_abab("a,d,b,c") * t2_abab("b,c,i,n");
  _a019_abab("j,m,i,n") +=  0.25 * _a004_abab("a,d,j,m") * t2_abab("a,d,i,n");
  i0_abab("a,c,i,n")    +=  4.0 * _a019_abab("j,m,i,n") * t2_abab("a,c,j,m");
  i0_abab("a,c,i,n")    += -1.0 * t2_abab("a,d,i,n") * _a001_bb("c,d");
  i0_abab("a,c,i,n")    += -1.0 * t2_abab("b,c,i,n") * _a001_aa("a,b");
  i0_abab("a,c,j,m")    += -1.0 * t2_abab("a,c,i,m") * _a006_aa("i,j");
  i0_abab("a,c,i,n")    += -1.0 * t2_abab("a,c,i,m") * _a006_bb("m,n");
  
  world.gop.fence();

  const double stop = madness::wall_time();
  const double cctime = stop - start;
  total_time += cctime;

  if (world.rank() == 0)
    std::cout << "Iteration " << i + 1 << "   time=" << cctime
              << "\n";
    
  }

  // Print results
  if (world.rank() == 0)
    std::cout << "Average CCSD per iteration time   = "
              << total_time / static_cast<double>(repeat - 1)
              << " sec" << "\n";

  world.gop.fence();

}

template <typename Tile, typename Policy>
void rand_fill_array(TA::DistArray<Tile, Policy>& array) {
  auto& world = array.world();
  // Iterate over local, non-zero tiles
  for (auto it : array) {
    // Construct a new tile with random data
    typename TA::DistArray<Tile, Policy>::value_type tile(
        array.trange().make_tile_range(it.index()));
    for (auto& tile_it : tile) tile_it = world.drand();

    // Set array tile
    it = tile;
  }
}
