#include "n8_binary_format.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
using u8=std::uint8_t;using u64=std::uint64_t;using i128=__int128_t;
constexpr int N=8,M=8;
struct Mat{std::array<u8,N>row{};std::array<u8,M>col{};};
static Mat decode(u64 key){Mat A;for(int i=N-1;i>=0;--i){A.row[i]=u8(key&255u);key>>=8;}for(int c=0;c<M;++c)for(int i=0;i<N;++i)if((A.row[i]>>c)&1u)A.col[c]|=u8(1u<<i);return A;}
static long long det(std::array<std::array<long long,M>,M>a){i128 prev=1;int sign=1;for(int c=0;c<M-1;++c){int p=c;while(p<M&&a[p][c]==0)++p;if(p==M)return 0;if(p!=c){std::swap(a[p],a[c]);sign=-sign;}i128 piv=a[c][c];for(int i=c+1;i<M;++i)for(int j=c+1;j<M;++j){i128 z=i128(a[i][j])*piv-i128(a[i][c])*a[c][j];if(z%prev!=0){std::cerr<<"Bareiss nondivision\n";std::abort();}z/=prev;if(z<std::numeric_limits<long long>::min()||z>std::numeric_limits<long long>::max()){std::cerr<<"overflow\n";std::abort();}a[i][j]=(long long)z;}prev=piv;}return sign*a[M-1][M-1];}
int main(int argc,char**argv){if(argc!=2){std::cerr<<"usage verify_n8m8_kernel_list n8m8_pos.bin\n";return 2;}std::ifstream f(argv[1],std::ios::binary);u64 n=0;f.read((char*)&n,8);if(!f){std::cerr<<"read header fail\n";return 1;}u64 prev=0;bool first=true;long long maxD=0,maxNum=0,minAbsD=std::numeric_limits<long long>::max();
 for(u64 ix=0;ix<n;++ix){u64 key;f.read((char*)&key,8);if(!f){std::cerr<<"truncated at "<<ix<<"\n";return 1;}if(!first&&key<=prev){std::cerr<<"not strictly sorted at "<<ix<<"\n";return 1;}first=false;prev=key;auto A=decode(key);for(int c=0;c<M;++c){if(A.col[c]==0||A.col[c]==255){std::cerr<<"trivial support at "<<ix<<"\n";return 1;}for(int d=c+1;d<M;++d){u8 x=A.col[c],y=A.col[d];if((x&y)==x||(x&y)==y){std::cerr<<"not antichain at "<<ix<<"\n";return 1;}}}
  std::array<std::array<long long,M>,M>B{};for(int c=0;c<M;++c)for(int i=0;i<N;++i)B[c][i]=(A.row[i]>>c)&1u;long long D=det(B);if(D==0){std::cerr<<"singular at "<<ix<<"\n";return 1;}maxD=std::max(maxD,std::llabs(D));minAbsD=std::min(minAbsD,std::llabs(D));for(int j=0;j<N;++j){auto C=B;for(int r=0;r<M;++r)C[r][j]=1;long long z=det(C);if((D>0&&z<=0)||(D<0&&z>=0)){std::cerr<<"nonpositive dual at "<<ix<<" voter "<<j<<" D="<<D<<" z="<<z<<"\n";return 1;}maxNum=std::max(maxNum,std::llabs(z));}
  if((ix+1)%1000000==0)std::cerr<<"\r"<<(ix+1)<<"/"<<n<<std::flush;
 }
 char extra;if(f.read(&extra,1)){std::cerr<<"trailing bytes\n";return 1;}std::cerr<<"\n";std::cout<<"PASS kernels="<<n<<" sorted_unique=1 antichain=1 full_rank=1 positive_dual=1 max_abs_det="<<maxD<<" max_abs_cramer="<<maxNum<<" min_abs_det="<<minAbsD<<"\n";}
