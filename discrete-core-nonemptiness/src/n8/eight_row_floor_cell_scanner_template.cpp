#include "n8_binary_format.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <filesystem>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <limits>
#include <mutex>
#include <numeric>
#include <string>
#include <thread>
#include <unordered_set>
#include <vector>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>

using u8 = std::uint8_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
#ifndef MM
#define MM 8
#endif
constexpr int N=8, M=MM;

struct Matrix { std::array<u8,M> col{}; std::array<u8,N> row{}; };
static Matrix decode_key(u64 key){
    Matrix A; constexpr u64 mask=(1u<<M)-1;
    std::array<u8,N> rows{};
    for(int i=N-1;i>=0;--i){rows[i]=u8(key&mask);key>>=M;}
    A.row=rows;
    for(int c=0;c<M;++c)for(int i=0;i<N;++i)if((rows[i]>>c)&1u)A.col[c]|=u8(1u<<i);
    return A;
}
static std::vector<u64> load_keys(const std::string& path){
    std::ifstream f(path,std::ios::binary);if(!f)throw std::runtime_error("cannot open "+path);
    u64 n=0;f.read(reinterpret_cast<char*>(&n),8);if(!f)throw std::runtime_error("truncated key header: "+path);
    const auto expected=std::uintmax_t(8)+std::uintmax_t(8)*n;
    if(std::filesystem::file_size(path)!=expected)throw std::runtime_error("key-list length mismatch: "+path);
    std::vector<u64>v(static_cast<std::size_t>(n));
    if(!v.empty()){f.read(reinterpret_cast<char*>(v.data()),static_cast<std::streamsize>(8*n));if(!f)throw std::runtime_error("truncated key payload: "+path);}
    return v;
}
static inline u32 pack(const std::array<u8,N>& a){u32 x=0;for(int i=0;i<N;++i)x|=u32(a[i])<<(3*i);return x;}
static inline std::array<u8,N> unpack(u32 x){std::array<u8,N>a{};for(int i=0;i<N;++i)a[i]=u8((x>>(3*i))&7u);return a;}
static inline bool leq(u32 a,u32 b){for(int i=0;i<N;++i)if(((a>>(3*i))&7u)>((b>>(3*i))&7u))return false;return true;}
static inline u8 coord(u32 x,int i){return u8((x>>(3*i))&7u);}
static inline u32 setcoord(u32 x,int i,u8 v){return (x&~(u32(7)<<(3*i)))|(u32(v)<<(3*i));}

// Two-phase simplex, adapted to tiny dense LPs: maximize c*x, A*x<=b, x>=0.
struct LPSolver {
    static constexpr double EPS=1e-10, INF=1e100;
    int m,n; std::vector<int>B,Nv; std::vector<std::vector<double>>D;
    LPSolver(const std::vector<std::vector<double>>&A,const std::vector<double>&b,const std::vector<double>&c)
      :m((int)b.size()),n((int)c.size()),B(m),Nv(n+1),D(m+2,std::vector<double>(n+2)){
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)D[i][j]=A[i][j];
        for(int i=0;i<m;++i){B[i]=n+i;D[i][n]=-1;D[i][n+1]=b[i];}
        for(int j=0;j<n;++j){Nv[j]=j;D[m][j]=-c[j];}
        Nv[n]=-1;D[m+1][n]=1;
    }
    void Pivot(int r,int s){
        double inv=1.0/D[r][s];
        for(int i=0;i<m+2;++i)if(i!=r)for(int j=0;j<n+2;++j)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;
        for(int j=0;j<n+2;++j)if(j!=s)D[r][j]*=inv;
        for(int i=0;i<m+2;++i)if(i!=r)D[i][s]*=-inv;
        D[r][s]=inv;std::swap(B[r],Nv[s]);
    }
    bool Simplex(int phase){
        int x=(phase==1?m+1:m);
        for(;;){
            int s=-1;for(int j=0;j<=n;++j){if(phase==2&&Nv[j]==-1)continue;if(s==-1||D[x][j]<D[x][s]-EPS||(std::abs(D[x][j]-D[x][s])<=EPS&&Nv[j]<Nv[s]))s=j;}
            if(D[x][s]>=-EPS)return true;
            int r=-1;for(int i=0;i<m;++i){if(D[i][s]<=EPS)continue;if(r==-1)r=i;else{double lhs=D[i][n+1]/D[i][s],rhs=D[r][n+1]/D[r][s];if(lhs<rhs-EPS||(std::abs(lhs-rhs)<=EPS&&B[i]<B[r]))r=i;}}
            if(r==-1){return false;}
            Pivot(r,s);
        }
    }
    double Solve(std::vector<double>*xout=nullptr){
        int r=0;for(int i=1;i<m;++i)if(D[i][n+1]<D[r][n+1])r=i;
        if(D[r][n+1]<-EPS){Pivot(r,n);if(!Simplex(1)||D[m+1][n+1]<-EPS)return -INF;if(D[m+1][n+1]<EPS){if(auto it=std::find(B.begin(),B.end(),-1);it!=B.end()){r=int(it-B.begin());int s=0;for(int j=1;j<=n;++j)if(D[r][j]<D[r][s]-EPS||(std::abs(D[r][j]-D[r][s])<=EPS&&Nv[j]<Nv[s]))s=j;Pivot(r,s);}}}
        if(!Simplex(2))return INF;
        if(xout){xout->assign(n,0);for(int i=0;i<m;++i)if(B[i]<n)(*xout)[B[i]]=D[i][n+1];}
        return D[m][n+1];
    }
};


using Big=long long;
using QR=boost::rational<Big>;
struct ExactLPSolver {
    int m,n;std::vector<int>B,Nv;std::vector<std::vector<QR>>D;
    ExactLPSolver(const std::vector<std::vector<long long>>&A,const std::vector<long long>&b,const std::vector<long long>&c)
      :m((int)b.size()),n((int)c.size()),B(m),Nv(n+1),D(m+2,std::vector<QR>(n+2)){
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)D[i][j]=QR(A[i][j]);
        for(int i=0;i<m;++i){B[i]=n+i;D[i][n]=-1;D[i][n+1]=QR(b[i]);}
        for(int j=0;j<n;++j){Nv[j]=j;D[m][j]=-QR(c[j]);}
        Nv[n]=-1;D[m+1][n]=1;
    }
    void Pivot(int r,int s){
        QR inv=QR(1)/D[r][s];
        for(int i=0;i<m+2;++i)if(i!=r)for(int j=0;j<n+2;++j)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;
        for(int j=0;j<n+2;++j)if(j!=s)D[r][j]*=inv;
        for(int i=0;i<m+2;++i)if(i!=r)D[i][s]*=-inv;
        D[r][s]=inv;std::swap(B[r],Nv[s]);
    }
    bool Simplex(int phase){
        int x=(phase==1?m+1:m);
        for(;;){
            int s=-1;
            for(int j=0;j<=n;++j){if(phase==2&&Nv[j]==-1)continue;if(s==-1||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&Nv[j]<Nv[s]))s=j;}
            if(D[x][s]>=0)return true;
            int r=-1;
            for(int i=0;i<m;++i){if(D[i][s]<=0)continue;if(r==-1)r=i;else{QR lhs=D[i][n+1]/D[i][s],rhs=D[r][n+1]/D[r][s];if(lhs<rhs||(lhs==rhs&&B[i]<B[r]))r=i;}}
            if(r==-1)return false;
            Pivot(r,s);
        }
    }
    // -1 infeasible, 0 optimal, 1 unbounded.
    int Solve(QR&value){
        int r=0;for(int i=1;i<m;++i)if(D[i][n+1]<D[r][n+1])r=i;
        if(D[r][n+1]<0){
            Pivot(r,n);
            if(!Simplex(1)||D[m+1][n+1]<0)return -1;
            if(D[m+1][n+1]!=0){std::cerr<<"phase-I sign anomaly\n";std::abort();}
            auto it=std::find(B.begin(),B.end(),-1);
            if(it!=B.end()){
                r=int(it-B.begin());int ss=-1;
                for(int j=0;j<=n;++j)if(D[r][j]!=0&&(ss==-1||D[r][j]<D[r][ss]||(D[r][j]==D[r][ss]&&Nv[j]<Nv[ss])))ss=j;
                if(ss!=-1)Pivot(r,ss);
            }
        }
        if(!Simplex(2))return 1;
        value=D[m][n+1];return 0;
    }
};

static std::vector<u32> utilities(const Matrix&A,int k){
    std::vector<u32> U;for(int z=0;z<(1<<M);++z)if(__builtin_popcount((unsigned)z)==k){std::array<u8,N>u{};for(int i=0;i<N;++i)u[i]=u8(__builtin_popcount((unsigned)(A.row[i]&z)));U.push_back(pack(u));}return U;
}

static std::vector<u32> minimal_holes(const std::vector<u32>&U,const std::array<u8,N>&upper){
    std::vector<u32> H{0};
    for(u32 u:U){
        std::vector<u32>C;C.reserve(H.size()*N);
        for(u32 h:H){
            if(leq(h,u)){
                for(int i=0;i<N;++i){u8 v=u8(coord(u,i)+1);if(v<=upper[i])C.push_back(setcoord(h,i,v));}
            }else C.push_back(h);
        }
        std::sort(C.begin(),C.end());C.erase(std::unique(C.begin(),C.end()),C.end());
        std::sort(C.begin(),C.end(),[](u32 a,u32 b){int sa=0,sb=0;for(int i=0;i<N;++i){sa+=coord(a,i);sb+=coord(b,i);}return sa!=sb?sa<sb:a<b;});
        std::vector<u32> K;K.reserve(C.size());
        for(u32 x:C){bool dom=false;for(u32 y:K)if(leq(y,x)){dom=true;break;}if(!dom)K.push_back(x);}H.swap(K);
    }
    return H;
}

static bool implementable(u32 h,const std::vector<u32>&U){for(u32 u:U)if(leq(h,u))return true;return false;}
static u8 usable_mask(u32 h,const std::vector<u32>&U){
    u8 b=0;for(int i=0;i<N;++i)if(coord(h,i)>0){u32 p=setcoord(h,i,u8(coord(h,i)-1));if(implementable(p,U))b|=u8(1u<<i);}return b;
}
using SubsetBounds = std::array<std::array<u8,1<<N>,M+1>;
static SubsetBounds subset_bounds(const Matrix&A){
    SubsetBounds B{};
    for(int S=1;S<(1<<N);++S){
        std::array<u8,M> val{};
        for(int c=0;c<M;++c) val[c]=u8(__builtin_popcount((unsigned)(A.col[c]&S)));
        std::sort(val.begin(),val.end(),std::greater<u8>());
        int sum=0;
        for(int k=1;k<=M;++k){sum+=val[k-1];B[k][S]=u8(sum);}
    }
    return B;
}
static bool subset_pass(u32 h,const std::array<u8,1<<N>&rhs){
    std::array<u8,1<<N> lhs{};
    for(int S=1;S<(1<<N);++S){int b=__builtin_ctz((unsigned)S);lhs[S]=u8(lhs[S&(S-1)]+coord(h,b));if(lhs[S]>rhs[S])return false;}
    return true;
}

static bool cover_dual_excludes(u32 h,const Matrix&A,int k){
    // Maximize lambda*h - sum mu subject to A^T lambda - mu <= 1,
    // lambda,mu >=0. Any rationalized lambda gives a sound lower bound.
    const int V=N+M;
    std::vector<std::vector<double>> C(M,std::vector<double>(V));
    std::vector<double>b(M,1.0),obj(V);
    for(int i=0;i<N;++i)obj[i]=coord(h,i);
    for(int c=0;c<M;++c){obj[N+c]=-1;for(int i=0;i<N;++i)C[c][i]=(A.row[i]>>c)&1u;C[c][N+c]=-1;}
    std::vector<double>sol;LPSolver lp(C,b,obj);double val=lp.Solve(&sol);
    if(val<=k+1e-8)return false;
    constexpr long long Q=1000000LL;
    std::array<long long,N>num{};
    for(int i=0;i<N;++i){double x=(i<(int)sol.size()?sol[i]:0);if(x<0)x=0;num[i]=std::llround(x*Q);if(num[i]<0)num[i]=0;}
    __int128 v=0;
    for(int i=0;i<N;++i)v+=(__int128)coord(h,i)*num[i];
    for(int c=0;c<M;++c){long long s=0;for(int i=0;i<N;++i)if((A.row[i]>>c)&1u)s+=num[i];if(s>Q)v-=s-Q;}
    return v>(__int128)k*Q;
}


static bool cell_positive_exact(const Matrix&A,int k,u32 h,u8 surplusMask){
    // Variables y=(x0,...,x_{M-2},eps)>=0; xlast=k-sum x.
    constexpr int V=M;std::vector<std::vector<long long>>C;std::vector<long long>b;
    auto add=[&](std::array<long long,V>a,long long rhs){C.emplace_back(a.begin(),a.end());b.push_back(rhs);};
    for(int c=0;c<M-1;++c){std::array<long long,V>a{};a[c]=-1;a[V-1]=1;add(a,0);a={};a[c]=1;a[V-1]=1;add(a,1);}
    {std::array<long long,V>a{};for(int c=0;c<M-1;++c)a[c]=1;a[V-1]=1;add(a,k);}
    {std::array<long long,V>a{};for(int c=0;c<M-1;++c)a[c]=-1;a[V-1]=1;add(a,1-k);}
    for(int i=0;i<N;++i){
        int al=(A.row[i]>>(M-1))&1u;long long constant=al*k;std::array<long long,V>lo{},up{};
        for(int c=0;c<M-1;++c){long long q=int((A.row[i]>>c)&1u)-al;lo[c]=-q;up[c]=q;}
        if((surplusMask>>i)&1u)lo[V-1]=1;
        up[V-1]=1;add(lo,constant-coord(h,i));add(up,coord(h,i)+1-constant);
    }
    std::vector<long long>obj(V);obj[V-1]=1;ExactLPSolver lp(C,b,obj);QR val;int st=lp.Solve(val);return st==0&&val>0;
}

using Record = n8fmt::HardRecord;
struct Stats {u64 cases=0,raw=0,norm=0,min_subset_rej=0,min_cover_rej=0,upper=0,subset=0,floor_cover_rej=0,feasible=0,unresolved=0,bempty=0;Stats&operator+=(const Stats&o){cases+=o.cases;raw+=o.raw;norm+=o.norm;min_subset_rej+=o.min_subset_rej;min_cover_rej+=o.min_cover_rej;upper+=o.upper;subset+=o.subset;floor_cover_rej+=o.floor_cover_rej;feasible+=o.feasible;unresolved+=o.unresolved;bempty+=o.bempty;return *this;}};

int main(int argc,char**argv){
    if(argc<4){std::cerr<<"usage: eight_voter_cells input_pos.bin output.bin threads [limit] [offset]\n";return 2;}
    auto keys=load_keys(argv[1]);std::string outpath=argv[2];int T=std::stoi(argv[3]);if(T<1)throw std::invalid_argument("threads must be positive");size_t limit=(argc>=5?std::stoull(argv[4]):keys.size());size_t offset=(argc>=6?std::stoull(argv[5]):0);if(offset>keys.size())offset=keys.size();limit=std::min(limit,keys.size()-offset);
    std::vector<std::vector<Record>> recs(T);std::vector<std::array<Stats,M+1>> st(T);std::atomic<size_t>done=0,next{0};auto t0=std::chrono::steady_clock::now();
    auto work=[&](int tid){
        auto&rr=recs[tid];rr.reserve(limit/100+100);
        for(;;){size_t jj=next.fetch_add(1,std::memory_order_relaxed);if(jj>=limit)break;size_t ix=offset+jj;u64 key=keys[ix];Matrix A=decode_key(key);auto SB=subset_bounds(A);
            for(int k=2;k<=M-2;++k){Stats&s=st[tid][k];++s.cases;auto U=utilities(A,k);std::array<u8,N>L{},Up{};for(int i=0;i<N;++i){int d=__builtin_popcount((unsigned)A.row[i]);if(d==0)L[i]=Up[i]=0;else if(d==M)L[i]=Up[i]=u8(k);else{L[i]=u8(std::max(0,k-(M-d)));Up[i]=u8(std::min(d,k)-1);}}
                auto H=minimal_holes(U,Up);s.raw+=H.size();std::unordered_set<u32> floors;floors.reserve(H.size()*4+1);
                for(u32 g:H){auto a=unpack(g);bool ok=true;for(int i=0;i<N;++i){a[i]=std::max(a[i],L[i]);if(a[i]>Up[i]){ok=false;break;}}if(!ok)continue;u32 gn=pack(a);if(implementable(gn,U))continue;++s.norm;if(!subset_pass(gn,SB[k])){++s.min_subset_rej;continue;}if(cover_dual_excludes(gn,A,k)){++s.min_cover_rej;continue;}
                    std::array<u8,N>cur=a;std::function<void(int)>gen=[&](int i){if(i==N){floors.insert(pack(cur));return;}u8 old=cur[i];for(int v=old;v<=Up[i];++v){cur[i]=u8(v);gen(i+1);}cur[i]=old;};gen(0);
                }
                s.upper+=floors.size();for(u32 h:floors){if(!subset_pass(h,SB[k]))continue;++s.subset;if(cover_dual_excludes(h,A,k)){++s.floor_cover_rej;continue;}if(!cell_positive_exact(A,k,h,0))continue;++s.feasible;u8 B=usable_mask(h,U);if(cell_positive_exact(A,k,h,B)){++s.unresolved;if(B==0)++s.bempty;Record r{key,h,u8(k),B,u8(B==0?1:0),0,1.0};rr.push_back(r);}}
            }
            size_t d=++done;if(d%10000==0){double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::cerr<<"\r"<<d<<"/"<<limit<<" matrices, "<<std::fixed<<std::setprecision(0)<<d/sec<<"/s"<<std::flush;}
        }
    };
    std::vector<std::thread>ths;for(int t=0;t<T;++t)ths.emplace_back(work,t);for(auto&t:ths)t.join();std::cerr<<"\n";
    std::vector<Record>all;size_t nr=0;for(auto&v:recs)nr+=v.size();all.reserve(nr);for(auto&v:recs)all.insert(all.end(),v.begin(),v.end());std::sort(all.begin(),all.end(),[](const Record&a,const Record&b){if(a.key!=b.key)return a.key<b.key;if(a.k!=b.k)return a.k<b.k;return a.h<b.h;});
    std::ofstream f(outpath,std::ios::binary|std::ios::trunc);if(!f)throw std::runtime_error("cannot create "+outpath);u64 magic=n8fmt::hard_magic,count=all.size();f.write(reinterpret_cast<char*>(&magic),8);f.write(reinterpret_cast<char*>(&count),8);if(!all.empty())f.write(reinterpret_cast<char*>(all.data()),static_cast<std::streamsize>(all.size()*sizeof(Record)));if(!f)throw std::runtime_error("write failed for "+outpath);
    std::array<Stats,M+1>sum{};for(auto&ss:st)for(int k=2;k<=M-2;++k)sum[k]+=ss[k];
    std::cout<<"input="<<argv[1]<<" offset="<<offset<<" matrices="<<limit<<" records="<<all.size()<<"\n";
    for(int k=2;k<=M-2;++k){auto&s=sum[k];std::cout<<"k="<<k<<" cases="<<s.cases<<" raw_min_holes="<<s.raw<<" normalized_min="<<s.norm<<" min_subset_rej="<<s.min_subset_rej<<" min_cover_rej="<<s.min_cover_rej<<" upper_floors="<<s.upper<<" subset_pass="<<s.subset<<" floor_cover_rej="<<s.floor_cover_rej<<" cell_feasible="<<s.feasible<<" unresolved_by_tight="<<s.unresolved<<" B_empty="<<s.bempty<<"\n";}
    std::cout<<"seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"\n";
}
