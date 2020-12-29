#include <algorithm>
#include "SynthTurb3d_common.hpp"

namespace SynthTurb
{
  template<class real_t, int Nmodes, int Nwaves_max>
  class SynthTurb3d_periodic_box : public SynthTurb3d_common<real_t, Nmodes, Nwaves_max>
  {
    using parent_t = SynthTurb3d_common<real_t, Nmodes, Nwaves_max>;

    int enm[3][Nmodes][Nwaves_max];
    int nn[Nmodes];

    void generate_wavenumbers(const real_t &Lmax, const real_t &Lmin) override
    {
      // wavevectors in the form k = (nx,ny,nz) * 2 PI / L, where n is integer to get periodic flow 

      // nn = nx^2 + ny^2 + nz^2 

      // linear distribution of nn (nn = 1, 2, 3, 4, ..., Nmodes)
   //   for(int n=0; n<Nmodes; ++n)
     //   this->k[n] = sqrt(n+1) * (2. * M_PI / Lmax);

      // geometric distribution of nn
      this->k[0] = 2. * M_PI / Lmax;
      nn[0]=1;

      real_t alpha = pow(Lmax / Lmin, 1. / (Nmodes - 1));
      while(1)
      {
//        std::cerr << "alpha: " << alpha << std::endl;

        for(int n=1; n<Nmodes; ++n)
        {
          nn[n] = -1;
          int exponent = n;
          while(nn[n] <= nn[n-1])
          {
            nn[n] = std::round(std::pow(alpha, exponent++));
          }
          if(nn[n] > Lmax / Lmin) break;
        }
        if(nn[Nmodes-1] <= Lmax / Lmin)
          break;
        else
          alpha /= 1.001;
      }

      for(int n=1; n<Nmodes; ++n)
      {
        std::cerr << "nn[" << n << "]: " << nn[n] << std::endl;
        this->k[n] = this->k[0] * sqrt(nn[n]);
      }


      std::vector<std::array<int,3>> vectors;
      for(int n=0; n<Nmodes; ++n)
      {
        this->Nwaves[n] = degeneracy_generator(nn[n], vectors);

        if(this->Nwaves[n] > Nwaves_max) // random shuffle, because not all possible degeneracies will be used
        {
          std::default_random_engine local_rand_eng(std::random_device{}());
          std::shuffle(std::begin(vectors), std::end(vectors), local_rand_eng);
          this->Nwaves[n] = Nwaves_max;
        }
      //  if(Nwaves_max != 6) throw std::runtime_error("nwaves max needs to be 6 for this test");
      //  vectors = {{1,0,0},{0,1,0},{0,0,1}};

        for(int m=0; m<this->Nwaves[n]; m+=2)
        {
          enm[0][n][m] = vectors.at(m/2)[0];
          enm[1][n][m] = vectors.at(m/2)[1];
          enm[2][n][m] = vectors.at(m/2)[2];
          // opposite vector
          enm[0][n][m+1] = -vectors.at(m/2)[0];
          enm[1][n][m+1] = -vectors.at(m/2)[1];
          enm[2][n][m+1] = -vectors.at(m/2)[2];
        }
      }
    }

    void generate_unit_wavevectors(const int &mode_idx, const int &wave_idx) override
    {
      this->e[0]=enm[0][mode_idx][wave_idx] / sqrt(mode_idx+1);
      this->e[1]=enm[1][mode_idx][wave_idx] / sqrt(mode_idx+1);
      this->e[2]=enm[2][mode_idx][wave_idx] / sqrt(mode_idx+1);
    }

    public:

    void update_time(const real_t &dt) override
    {
      #pragma omp parallel for
      for(int n=0; n<Nmodes; ++n)
      {
        std::normal_distribution<real_t> normal_d(0,1);
        std::default_random_engine local_rand_eng(std::random_device{}());
        real_t relax = exp(-this->wn[n] * dt);

        for(int m=0; m<this->Nwaves[n]; m+=2)
        {
          for(int i=0; i<3; ++i)
          {
            this->Anm[i][n][m] = relax * this->Anm[i][n][m] + this->std_dev[n] * sqrt(1. - relax * relax) * normal_d(local_rand_eng);
            this->Anm[i][n][m+1] = -this->Anm[i][n][m];
        //    this->Anm[i][n][m+1] = relax * this->Anm[i][n][m+1] + this->std_dev[n] * sqrt(1. - relax * relax) * normal_d(local_rand_eng);

            this->Bnm[i][n][m] = relax * this->Bnm[i][n][m] + this->std_dev[n] * sqrt(1. - relax * relax) * normal_d(local_rand_eng);
            this->Bnm[i][n][m+1] = this->Bnm[i][n][m];
        //    this->Bnm[i][n][m+1] = relax * this->Bnm[i][n][m+1] + this->std_dev[n] * sqrt(1. - relax * relax) * normal_d(local_rand_eng);
          }
        }
      }
    }

    public:

    //ctor
    SynthTurb3d_periodic_box(
      const real_t &eps,        // TKE dissipation rate [m2/s3]
      const real_t &Lmax = 100, // maximum length scale [m]
      const real_t &Lmin = 1e-3 // Kolmogorov length scale [m]
    )
    {
      if(Nwaves_max % 2 != 0) throw std::runtime_error("Nwaves_max needs to be even, because we need to include opposites of all wavevectors.");
      this->init(eps, Lmax, Lmin);
    }
  };
};
