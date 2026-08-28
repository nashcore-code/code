#include "n8_binary_format.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <boost/rational.hpp>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <iostream>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <string>
#include <vector>
#ifndef MM
#define MM 8
#endif
constexpr int N=8,M=MM;
using u8=std::uint8_t;using u32=std::uint32_t;using u64=std::uint64_t;
using Rat=boost::rational<long long>;

struct Matrix{std::array<u8,M>col{};std::array<u8,N>row{};};
static Matrix decode_key(u64 key){Matrix A;u64 mask=(u64(1)<<M)-1;for(int i=N-1;i>=0;--i){A.row[i]=u8(key&mask);key>>=M;}for(int c=0;c<M;++c)for(int i=0;i<N;++i)if((A.row[i]>>c)&1u)A.col[c]|=u8(1u<<i);return A;}
static bool antichain_kernel(const Matrix&A){for(int c=0;c<M;++c)for(int d=c+1;d<M;++d){unsigned x=A.col[c],y=A.col[d];if((x&y)==x||(x&y)==y)return false;}return true;}
using Record = n8fmt::HardRecord;
using Out = n8fmt::CertificateRecord;
static int hc(u32 h,int i){return int((h>>(3*i))&7u);}
static std::array<int,N> unpack(u32 h){std::array<int,N>a{};for(int i=0;i<N;++i)a[i]=hc(h,i);return a;}
static std::array<int,N> util(const Matrix&A,int cm){std::array<int,N>q{};for(int i=0;i<N;++i)q[i]=std::popcount((unsigned)(A.row[i]&cm));return q;}
static void verify_puncture(const Matrix&A,const std::array<int,N>&h,int k,int cm,int i){
    constexpr unsigned valid_mask=(1u<<M)-1u;
    if((unsigned(cm)&~valid_mask)!=0u){std::cerr<<"committee mask fail\n";std::abort();}
    if(std::popcount((unsigned)cm)!=k){std::cerr<<"committee size fail\n";std::abort();}
    auto q=util(A,cm);
    if(q[i]!=h[i]-1){std::cerr<<"genuine deficit fail\n";std::abort();}
    for(int j=0;j<N;++j)if(j!=i&&q[j]<h[j]){std::cerr<<"puncture cover fail\n";std::abort();}
}

static Rat sat_lower_exact(int k,int s,int d){
    Rat best(1,s);
    for(int K=k;K<k+8;++K){
        int den=8*((K*s)/8+d);
        if(den>0){Rat z(K,den);if(z<best)best=z;}
    }
    return best;
}


struct ExactLP{
    int m,n;std::vector<int>B,Nv;std::vector<std::vector<Rat>>D;
    ExactLP(const std::vector<std::vector<long long>>&A,const std::vector<long long>&b,const std::vector<long long>&c):m((int)b.size()),n((int)c.size()),B(m),Nv(n+1),D(m+2,std::vector<Rat>(n+2)){
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)D[i][j]=Rat(A[i][j]);
        for(int i=0;i<m;++i){B[i]=n+i;D[i][n]=-1;D[i][n+1]=Rat(b[i]);}
        for(int j=0;j<n;++j){Nv[j]=j;D[m][j]=-Rat(c[j]);}Nv[n]=-1;D[m+1][n]=1;
    }
    void pivot(int r,int s){Rat inv=Rat(1)/D[r][s];for(int i=0;i<m+2;++i)if(i!=r)for(int j=0;j<n+2;++j)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;for(int j=0;j<n+2;++j)if(j!=s)D[r][j]*=inv;for(int i=0;i<m+2;++i)if(i!=r)D[i][s]*=-inv;D[r][s]=inv;std::swap(B[r],Nv[s]);}
    bool simplex(int ph){int x=(ph==1?m+1:m);for(;;){int s=-1;for(int j=0;j<=n;++j){if(ph==2&&Nv[j]==-1)continue;if(s==-1||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&Nv[j]<Nv[s]))s=j;}if(D[x][s]>=0)return true;int r=-1;for(int i=0;i<m;++i){if(D[i][s]<=0)continue;if(r==-1)r=i;else{Rat l=D[i][n+1]/D[i][s],z=D[r][n+1]/D[r][s];if(l<z||(l==z&&B[i]<B[r]))r=i;}}if(r==-1)return false;pivot(r,s);}}
    Rat solve(){int r=0;for(int i=1;i<m;++i)if(D[i][n+1]<D[r][n+1])r=i;if(D[r][n+1]<0){pivot(r,n);if(!simplex(1)||D[m+1][n+1]<0){std::cerr<<"certificate cell infeasible\n";std::abort();}auto it=std::find(B.begin(),B.end(),-1);if(it!=B.end()){r=int(it-B.begin());int s=-1;for(int j=0;j<=n;++j)if(D[r][j]!=0&&(s==-1||D[r][j]<D[r][s]||(D[r][j]==D[r][s]&&Nv[j]<Nv[s])))s=j;if(s!=-1)pivot(r,s);}}if(!simplex(2)){std::cerr<<"unbounded cell objective\n";std::abort();}return D[m][n+1];}
};

struct Cell{
    std::vector<std::vector<long long>>C;std::vector<long long>b;int k;
    std::map<std::array<long long,M>,Rat>cache;
    Cell(const Matrix&A,int kk,const std::array<int,N>&h):k(kk){
        int V=M-1;auto add=[&](std::vector<long long>a,long long z){C.push_back(std::move(a));b.push_back(z);};
        for(int c=0;c<V;++c){std::vector<long long>a(V);a[c]=1;add(a,1);}
        {std::vector<long long>a(V,1);add(a,k);}
        {std::vector<long long>a(V,-1);add(a,1-k);}
        for(int i=0;i<N;++i){int al=(A.row[i]>>(M-1))&1u;long long con=al*k;std::vector<long long>lo(V),up(V);for(int c=0;c<V;++c){long long q=int((A.row[i]>>c)&1u)-al;lo[c]=-q;up[c]=q;}add(lo,con-h[i]);add(up,h[i]+1-con);}
    }
    Rat maxdot(const std::array<long long,M>&p){
        auto it=cache.find(p);if(it!=cache.end())return it->second;
        int V=M-1;std::vector<long long>obj(V);for(int c=0;c<V;++c)obj[c]=p[c]-p[M-1];ExactLP lp(C,b,obj);Rat z=Rat(p[M-1]*k)+lp.solve();cache.emplace(p,z);return z;
    }
};

struct Aff{int dim=0;bool ok=false;std::array<Rat,N>a0{},u{};};
static Aff affine_dual(const Matrix&A){
    std::vector<std::vector<Rat>>R(M,std::vector<Rat>(N+1));
    for(int c=0;c<M;++c){for(int i=0;i<N;++i)R[c][i]=Rat((A.row[i]>>c)&1u);R[c][N]=1;}
    std::vector<int>piv;int rank=0;
    for(int col=0;col<N&&rank<M;++col){int p=rank;while(p<M&&R[p][col]==0)++p;if(p==M)continue;std::swap(R[p],R[rank]);Rat z=R[rank][col];for(int j=col;j<=N;++j)R[rank][j]/=z;for(int r=0;r<M;++r)if(r!=rank&&R[r][col]!=0){Rat f=R[r][col];for(int j=col;j<=N;++j)R[r][j]-=f*R[rank][j];}piv.push_back(col);++rank;}
    Aff P;if(rank!=M)return P;P.dim=N-M;if(P.dim>1)return P;std::array<bool,N>is{};for(int x:piv)is[x]=true;std::vector<int>fr;for(int i=0;i<N;++i)if(!is[i])fr.push_back(i);for(int r=0;r<M;++r)P.a0[piv[r]]=R[r][N];if(P.dim==1){P.u[fr[0]]=1;for(int r=0;r<M;++r)P.u[piv[r]]=-R[r][fr[0]];}P.ok=true;return P;
}
static std::string ak(const std::array<Rat,N>&a){std::ostringstream o;for(auto&x:a)o<<x.numerator()<<'/'<<x.denominator()<<';';return o.str();}
static std::vector<std::array<Rat,N>> dverts(const Aff&P,const std::array<Rat,N>&lb){
    if(!P.ok)return {};if(P.dim==0){for(int i=0;i<N;++i)if(P.a0[i]<lb[i])return {};return {P.a0};}
    bool hlo=false,hhi=false;Rat lo,hi;
    for(int i=0;i<N;++i){if(P.u[i]==0){if(P.a0[i]<lb[i])return {};}else{Rat z=(lb[i]-P.a0[i])/P.u[i];if(P.u[i]>0){if(!hlo||z>lo){lo=z;hlo=true;}}else{if(!hhi||z<hi){hi=z;hhi=true;}}}}
    if(hlo&&hhi&&lo>hi)return {};
    std::vector<std::array<Rat,N>>out;std::set<std::string>seen;
    auto add=[&](Rat t){std::array<Rat,N>a{};for(int i=0;i<N;++i)a[i]=P.a0[i]+P.u[i]*t;auto s=ak(a);if(seen.insert(s).second)out.push_back(a);};
    if(hlo)add(lo);if(hhi)add(hi);if(out.empty()){std::cerr<<"line has no finite vertex\n";std::abort();}return out;
}
struct Scaled{long long den=1;std::array<long long,N>num{};};
static Scaled scale(const std::array<Rat,N>&a){
    Scaled z;for(auto&x:a)z.den=std::lcm(z.den,x.denominator());for(int i=0;i<N;++i)z.num[i]=a[i].numerator()*(z.den/a[i].denominator());return z;
}
static std::vector<Rat> singleton_vals(const Matrix&A,const std::array<int,N>&h,int k,int i,const Aff&P,Cell&X){
    std::array<Rat,N>lb{};auto D=dverts(P,lb);std::vector<Rat> vals;
    for(auto&a:D){auto s=scale(a);std::array<long long,M>p{};for(int c=0;c<M;++c)p[c]=s.num[i]*((A.row[i]>>c)&1u);Rat z=Rat(k,8)*Rat(s.den-s.num[i])+Rat(s.num[i]*h[i])-X.maxdot(p);vals.push_back(z/Rat(s.den));}
    return vals;
}
static std::vector<Rat> adaptive_vals(const Matrix&A,const std::array<int,N>&h,int k,int E,const Aff&P,Cell&X){
    std::array<Rat,N>lb{};auto D=dverts(P,lb);std::vector<Rat>vals;int es=std::popcount((unsigned)E);
    for(auto&a:D){auto s=scale(a);long long sum=0,hh=0;std::array<long long,M>p{};for(int i=0;i<N;++i)if((E>>i)&1u){sum+=s.num[i];hh+=s.num[i]*h[i];for(int c=0;c<M;++c)if((A.row[i]>>c)&1u)p[c]+=s.num[i];}Rat z=Rat(k,8)*Rat(es*s.den-sum)+Rat(hh)-X.maxdot(p);vals.push_back(z/Rat(s.den));}
    return vals;
}
static std::vector<Rat> coalvals(const Matrix&A,const std::array<int,N>&h,int k,int cm,int i,int S,const Aff&P,Cell&X){
    int ss=std::popcount((unsigned)S);auto q=util(A,cm);std::array<Rat,N>lb{};
    for(int j=0;j<N;++j)if((S>>j)&1u)lb[j]=sat_lower_exact(k,ss,h[j]-q[j]);
    auto D=dverts(P,lb);std::vector<Rat>vals;
    for(auto&a:D){auto z=scale(a);long long con=0;std::array<long long,M>p{};for(int j=0;j<N;++j)if((S>>j)&1u){con+=z.num[j]*(q[j]+1);for(int c=0;c<M;++c)if((A.row[j]>>c)&1u)p[c]+=z.num[j];}vals.push_back((Rat(con)-X.maxdot(p))/Rat(z.den));}
    return vals;
}
static std::string rs(const Rat&x){std::ostringstream o;o<<x.numerator()<<"/"<<x.denominator();return o.str();}


static bool strict_line_safe(const std::vector<Rat>&v,bool&boundary,Rat&positive_measure){
    boundary=false;
    if(v.empty())return true;
    Rat mn=v[0],mx=v[0];for(auto&z:v){if(z<mn)mn=z;if(z>mx)mx=z;}
    if(v.size()==1){if(mn>0){positive_measure=mn;return true;}return false;}
    if(mn>=0&&mx>0){boundary=(mn==0);positive_measure=(mn>0?mn:mx);return true;}
    return false;
}


static bool coalition_safe_unique(const Matrix&A,const std::array<int,N>&h,int k,int cm,int S,
                                  const Aff&P,Cell&X,int&kind,Rat&positive){
    if(!P.ok||P.dim!=0){std::cerr<<"expected unique dual\n";std::abort();}
    auto q=util(A,cm);auto const&a=P.a0;int ss=std::popcount((unsigned)S);
    for(int j=0;j<N;++j)if((S>>j)&1u){
        Rat low=sat_lower_exact(k,ss,h[j]-q[j]);
        if(a[j]<=low){kind=0;return true;} // strict saturation bound makes S impossible
    }
    Rat crude=0;for(int j=0;j<N;++j)if((S>>j)&1u)crude+=a[j]*Rat(q[j]-h[j]);
    if(crude>=0){kind=1;positive=crude;return true;} // open-cell reserve is strictly positive
    auto z=scale(a);long long con=0;std::array<long long,M>p{};
    for(int j=0;j<N;++j)if((S>>j)&1u){
        con+=z.num[j]*(q[j]+1);
        for(int c=0;c<M;++c)if((A.row[j]>>c)&1u)p[c]+=z.num[j];
    }
    Rat v=(Rat(con)-X.maxdot(p))/Rat(z.den);
    kind=2;positive=v;return v>0;
}

int main(int argc,char**argv){
    if(argc<2){std::cerr<<"usage exact_cps_m8_checker certs.bin\n";return 2;}
    std::ifstream f(argv[1],std::ios::binary);if(!f)throw std::runtime_error(std::string("cannot open ")+argv[1]);u64 n=0;f.read(reinterpret_cast<char*>(&n),8);if(!f)throw std::runtime_error("truncated certificate header");const auto expected=std::uintmax_t(8)+std::uintmax_t(sizeof(Out))*n;if(std::filesystem::file_size(argv[1])!=expected)throw std::runtime_error("certificate length mismatch");std::vector<Out>V(static_cast<std::size_t>(n));if(!V.empty()){f.read(reinterpret_cast<char*>(V.data()),static_cast<std::streamsize>(V.size()*sizeof(Out)));if(!f)throw std::runtime_error("truncated certificate payload");}for(auto const&o:V){if(o.r.reserved!=0||!std::all_of(o.reserved.begin(),o.reserved.end(),[](auto x){return x==0;}))throw std::runtime_error("nonzero reserved bytes in certificate");if(o.type>1)throw std::runtime_error("unknown certificate type");}
    size_t ix=0;unsigned long long fixed=0,adapt=0,sat_skip=0,open_floor=0,lp_checks=0;
    Rat min_single,min_lp;bool hs=false,hl=false;
    for(auto const&o:V){++ix;auto A=decode_key(o.r.key);auto h=unpack(o.r.h);auto P=affine_dual(A);
        if(!P.ok||P.dim!=0){std::cerr<<"nonunique dual key "<<o.r.key<<"\n";return 1;}
        for(int i=0;i<N;++i)if(!(P.a0[i]>0)){std::cerr<<"nonpositive dual key "<<o.r.key<<" voter "<<i<<"\n";return 1;}
        for(int c=0;c<M;++c){Rat z=0;for(int i=0;i<N;++i)if((A.row[i]>>c)&1u)z+=P.a0[i];if(z!=1){std::cerr<<"dual equation fail key "<<o.r.key<<" column "<<c<<"\n";return 1;}}
        if(!antichain_kernel(A)){std::cerr<<"invalid antichain kernel key "<<o.r.key<<"\n";return 1;}
        if(o.r.k<2||o.r.k>M-2){std::cerr<<"invalid residual budget key "<<o.r.key<<" k "<<int(o.r.k)<<"\n";return 1;}
        if(o.type==0){
            if(o.deficit<0||o.deficit>=N||((o.r.Bmask>>o.deficit)&1u)==0){std::cerr<<"invalid fixed deficit key "<<o.r.key<<"\n";return 1;}
        }else{
            if(o.deficit!=-1||o.committee==0||((o.committee&~unsigned(o.r.Bmask))!=0)){std::cerr<<"invalid adaptive index set key "<<o.r.key<<"\n";return 1;}
        }
        Cell X(A,o.r.k,h);
        if(o.type==0){
            ++fixed;verify_puncture(A,h,o.r.k,o.committee,o.deficit);
            auto sv=singleton_vals(A,h,o.r.k,o.deficit,P,X);
            if(sv.size()!=1||!(sv[0]>0)){std::cerr<<"singleton fail key "<<o.r.key<<"\n";return 1;}
            if(!hs||sv[0]<min_single){min_single=sv[0];hs=true;}
            for(int S=1;S<256;++S)if((S>>o.deficit)&1u&&std::popcount((unsigned)S)>=2){
                int kind=-1;Rat z;
                if(!coalition_safe_unique(A,h,o.r.k,o.committee,S,P,X,kind,z)){
                    std::cerr<<"coal fail key "<<o.r.key<<" S "<<S<<" margin "<<rs(z)<<"\n";return 1;
                }
                if(kind==0)++sat_skip;else if(kind==1)++open_floor;else{++lp_checks;if(!hl||z<min_lp){min_lp=z;hl=true;}}
            }
        }else if(o.type==1){
            ++adapt;int E=o.committee;
            for(int i=0;i<N;++i)if((E>>i)&1u){
                verify_puncture(A,h,o.r.k,o.allcm[i],i);
                for(int S=1;S<256;++S)if((S>>i)&1u&&std::popcount((unsigned)S)>=2){
                    int kind=-1;Rat z;
                    if(!coalition_safe_unique(A,h,o.r.k,o.allcm[i],S,P,X,kind,z)){
                        std::cerr<<"adaptive coal fail key "<<o.r.key<<" i "<<i<<" S "<<S<<"\n";return 1;
                    }
                    if(kind==0)++sat_skip;else if(kind==1)++open_floor;else{++lp_checks;if(!hl||z<min_lp){min_lp=z;hl=true;}}
                }
            }
            auto av=adaptive_vals(A,h,o.r.k,E,P,X);
            if(av.size()!=1||!(av[0]>0)){std::cerr<<"adaptive sum fail key "<<o.r.key<<"\n";return 1;}
            if(!hs||av[0]<min_single){min_single=av[0];hs=true;}
        }else{std::cerr<<"unknown type\n";return 1;}
        if(ix%1000==0)std::cerr<<"\r"<<ix<<"/"<<n<<std::flush;
    }
    std::cerr<<"\n";
    std::cout<<"PASS certs="<<n<<" fixed="<<fixed<<" adaptive="<<adapt
      <<" saturation_skips="<<sat_skip<<" open_floor_checks="<<open_floor
      <<" exact_price_LPs="<<lp_checks
      <<" min_singleton_or_sum="<<(hs?rs(min_single):std::string("NA"))
      <<" min_exact_price="<<(hl?rs(min_lp):std::string("NA"))<<"\n";
}
