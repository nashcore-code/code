#include "n8_binary_format.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <filesystem>
#include <stdexcept>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <string>
#include <vector>

#ifndef MM
#define MM 8
#endif
constexpr int N=8,M=MM;
using u8=std::uint8_t;using u32=std::uint32_t;using u64=std::uint64_t;

struct Matrix{std::array<u8,M>col{};std::array<u8,N>row{};};
static Matrix decode_key(u64 key){Matrix A;u64 mask=(u64(1)<<M)-1;for(int i=N-1;i>=0;--i){A.row[i]=u8(key&mask);key>>=M;}for(int c=0;c<M;++c)for(int i=0;i<N;++i)if((A.row[i]>>c)&1u)A.col[c]|=u8(1u<<i);return A;}

struct LPSolver {
    static constexpr double EPS=1e-10,INF=1e100;
    int m,n;std::vector<int>B,Nv;std::vector<std::vector<double>>D;
    LPSolver(const std::vector<std::vector<double>>&A,const std::vector<double>&b,const std::vector<double>&c):m((int)b.size()),n((int)c.size()),B(m),Nv(n+1),D(m+2,std::vector<double>(n+2)){
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)D[i][j]=A[i][j];
        for(int i=0;i<m;++i){B[i]=n+i;D[i][n]=-1;D[i][n+1]=b[i];}
        for(int j=0;j<n;++j){Nv[j]=j;D[m][j]=-c[j];}Nv[n]=-1;D[m+1][n]=1;
    }
    void Pivot(int r,int s){double inv=1.0/D[r][s];for(int i=0;i<m+2;++i)if(i!=r)for(int j=0;j<n+2;++j)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;for(int j=0;j<n+2;++j)if(j!=s)D[r][j]*=inv;for(int i=0;i<m+2;++i)if(i!=r)D[i][s]*=-inv;D[r][s]=inv;std::swap(B[r],Nv[s]);}
    bool Simplex(int phase){int x=(phase==1?m+1:m);for(;;){int s=-1;for(int j=0;j<=n;++j){if(phase==2&&Nv[j]==-1)continue;if(s==-1||D[x][j]<D[x][s]-EPS||(std::abs(D[x][j]-D[x][s])<=EPS&&Nv[j]<Nv[s]))s=j;}if(D[x][s]>=-EPS)return true;int r=-1;for(int i=0;i<m;++i){if(D[i][s]<=EPS)continue;if(r==-1)r=i;else{double l=D[i][n+1]/D[i][s],z=D[r][n+1]/D[r][s];if(l<z-EPS||(std::abs(l-z)<=EPS&&B[i]<B[r]))r=i;}}if(r==-1)return false;Pivot(r,s);}}
    double Solve(){int r=0;for(int i=1;i<m;++i)if(D[i][n+1]<D[r][n+1])r=i;if(D[r][n+1]<-EPS){Pivot(r,n);if(!Simplex(1)||D[m+1][n+1]<-EPS)return -INF;if(D[m+1][n+1]<EPS){auto it=std::find(B.begin(),B.end(),-1);if(it!=B.end()){r=int(it-B.begin());int s=0;for(int j=1;j<=n;++j)if(D[r][j]<D[r][s]-EPS||(std::abs(D[r][j]-D[r][s])<=EPS&&Nv[j]<Nv[s]))s=j;Pivot(r,s);}}}if(!Simplex(2))return INF;return D[m][n+1];}
};

using Record = n8fmt::HardRecord;
static int coord(u32 h,int i){return int((h>>(3*i))&7u);}
static std::array<int,N> unpack(u32 h){std::array<int,N>a{};for(int i=0;i<N;++i)a[i]=coord(h,i);return a;}

struct DualLine{std::array<double,N>a0{},u{};int dim=0;bool ok=false;};
static DualLine dual_line(const Matrix&A){
    // RREF of A^T alpha=1, M equations, N variables.
    std::vector<std::vector<double>> R(M,std::vector<double>(N+1));
    for(int c=0;c<M;++c){for(int i=0;i<N;++i)R[c][i]=(A.row[i]>>c)&1u;R[c][N]=1;}
    std::vector<int> piv;int rank=0;
    for(int col=0;col<N&&rank<M;++col){int p=rank;while(p<M&&std::abs(R[p][col])<1e-12)++p;if(p==M)continue;std::swap(R[p],R[rank]);double z=R[rank][col];for(int j=col;j<=N;++j)R[rank][j]/=z;for(int r=0;r<M;++r)if(r!=rank&&std::abs(R[r][col])>1e-12){double f=R[r][col];for(int j=col;j<=N;++j)R[r][j]-=f*R[rank][j];}piv.push_back(col);++rank;}
    DualLine P;if(rank!=M)return P;std::array<bool,N>isp{};for(int x:piv)isp[x]=true;std::vector<int>free;for(int j=0;j<N;++j)if(!isp[j])free.push_back(j);P.dim=N-M;if(P.dim>1)return P;
    for(int r=0;r<M;++r)P.a0[piv[r]]=R[r][N];
    if(P.dim==1){int f=free[0];P.u[f]=1;for(int r=0;r<M;++r)P.u[piv[r]]=-R[r][f];}
    P.ok=true;return P;
}
static std::vector<std::array<double,N>> dual_vertices(const DualLine&P,const std::array<double,N>&lb){
    std::vector<std::array<double,N>> out;if(!P.ok)return out;
    if(P.dim==0){for(int i=0;i<N;++i)if(P.a0[i]<lb[i]-1e-9)return out;out.push_back(P.a0);return out;}
    double lo=-1e100,hi=1e100;
    for(int i=0;i<N;++i){double a=P.a0[i],u=P.u[i];if(std::abs(u)<1e-12){if(a<lb[i]-1e-9)return {};}else if(u>0)lo=std::max(lo,(lb[i]-a)/u);else hi=std::min(hi,(lb[i]-a)/u);}
    if(lo>hi+1e-9)return {};
    auto make=[&](double t){std::array<double,N>a{};for(int i=0;i<N;++i)a[i]=P.a0[i]+P.u[i]*t;return a;};
    if(lo>-1e90)out.push_back(make(lo));
    if(hi<1e90&& (out.empty()||hi-lo>1e-9))out.push_back(make(hi));
    if(out.empty()){std::cerr<<"unbounded line without endpoint\n";std::abort();}
    return out;
}

struct CellLP{
    std::vector<std::vector<double>>C;std::vector<double>b;int k;
    std::array<double,256> submax{};std::array<unsigned char,256> have{};
    CellLP(const Matrix&A,int kk,const std::array<int,N>&h):k(kk){
        int V=M-1;auto add=[&](std::vector<double>a,double rhs){C.push_back(std::move(a));b.push_back(rhs);};
        for(int c=0;c<V;++c){std::vector<double>a(V);a[c]=1;add(a,1);}
        {std::vector<double>a(V,1);add(a,k);}
        {std::vector<double>a(V,-1);add(a,1-k);}
        for(int i=0;i<N;++i){int al=(A.row[i]>>(M-1))&1u;double con=al*k;std::vector<double>lo(V),up(V);for(int c=0;c<V;++c){double q=int((A.row[i]>>c)&1u)-al;lo[c]=-q;up[c]=q;}add(lo,con-h[i]);add(up,h[i]+1-con);}
    }
    double max_dot(const std::array<double,M>&p){
        int V=M-1;std::vector<double>obj(V);for(int c=0;c<V;++c)obj[c]=p[c]-p[M-1];LPSolver lp(C,b,obj);double z=lp.Solve();if(z<=-LPSolver::INF/2){std::cerr<<"infeasible record cell\n";std::abort();}return p[M-1]*k+z;
    }
    double max_subset(const Matrix&A,const std::array<double,N>&a,int S){
        if(have[S])return submax[S];
        std::array<double,M>p{};for(int j=0;j<N;++j)if((S>>j)&1u)for(int c=0;c<M;++c)if((A.row[j]>>c)&1u)p[c]+=a[j];
        submax[S]=max_dot(p);have[S]=1;return submax[S];
    }
};

static std::array<int,N> util(const Matrix&A,int cm){std::array<int,N>q{};for(int i=0;i<N;++i)q[i]=std::popcount((unsigned)(A.row[i]&cm));return q;}
static bool puncture(const std::array<int,N>&q,const std::array<int,N>&h,int i){if(q[i]!=h[i]-1)return false;for(int j=0;j<N;++j)if(j!=i&&q[j]<h[j])return false;return true;}

static double sat_lower(int k,int s,int d){
    double best=1.0/s;
    for(int K=k;K<k+8;++K){
        int den=8*((K*s)/8+d);
        if(den>0) best=std::min(best,double(K)/double(den));
    }
    return best;
}

struct CheckResult{bool ok=false;double sg=-1e100,coal=1e100;int zeroS=0;};

// Proposal-side analogue of the exact checker's strict-line rule.  A margin
// on a one-point domain must be strictly positive.  On a nondegenerate dual
// segment, nonnegativity at both endpoints and strict positivity at at least
// one endpoint suffice because the floor cell is open.  The returned measure
// is only used to choose among proposals; exact replay recomputes everything.
static bool strict_line_safe(const std::vector<double>&v,double&positive_measure){
    if(v.empty()){positive_measure=1e100;return true;}
    double mn=*std::min_element(v.begin(),v.end());
    double mx=*std::max_element(v.begin(),v.end());
    if(v.size()==1){if(mn>1e-8){positive_measure=mn;return true;}return false;}
    if(mn>=-1e-8&&mx>1e-8){positive_measure=(mn>1e-8?mn:mx);return true;}
    return false;
}

static CheckResult check_committee(const Matrix&A,const std::array<int,N>&h,int k,int cm,int deficit,const DualLine&P,CellLP&X,bool needSingleton){
    CheckResult R;auto q=util(A,cm);if(!puncture(q,h,deficit))return R;
    std::array<double,N>zero{};auto base=dual_vertices(P,zero);if(base.empty())return R;

    // Preserve the audited square-kernel proposal path exactly.  For m=8 the
    // positive dual is unique, and the shortcuts below match the exact replay.
    if(P.dim==0){
        auto const&a=base[0];double sg=1e100;
        if(needSingleton){
            std::array<double,M>p{};for(int c=0;c<M;++c)p[c]=a[deficit]*((A.row[deficit]>>c)&1u);
            double maxr=X.max_dot(p);sg=(double(k)/8.0)*(1-a[deficit])+a[deficit]*h[deficit]-maxr;
            if(sg<=1e-8){R.sg=sg;return R;}
        }
        double worst=1e100;
        for(int S=1;S<256;++S)if((S>>deficit)&1u&&std::popcount((unsigned)S)>=2){
            int s=std::popcount((unsigned)S);bool possible=true;
            for(int j=0;j<N;++j)if((S>>j)&1u){
                double low=sat_lower(k,s,h[j]-q[j]);
                if(a[j]<=low+1e-10){possible=false;break;}
            }
            if(!possible)continue;
            // Since delta_j<1, a nonnegative floor-level margin is already
            // strictly positive in the actual open floor cell.
            double crude=0;for(int j=0;j<N;++j)if((S>>j)&1u)crude+=a[j]*(q[j]-h[j]);
            if(crude>=-1e-10){if(crude>1e-8)worst=std::min(worst,crude);continue;}
            double con=0;for(int j=0;j<N;++j)if((S>>j)&1u)con+=a[j]*(q[j]+1);
            double sm=con-X.max_subset(A,a,S);
            if(sm<worst)worst=sm;
            if(sm<=1e-8){R.sg=sg;R.coal=sm;R.zeroS=S;return R;}
        }
        R.ok=true;R.sg=needSingleton?sg:0;R.coal=worst;return R;
    }

    // For m=7 the dual domain is a line.  Saturation lower bounds cut out a
    // (possibly empty) subsegment for each coalition.  Evaluate the concave
    // floor-cell margin at every endpoint, exactly mirroring
    // exact_cps_fullsat_checker.cpp at proposal precision.
    double worst=1e100;
    for(int S=1;S<256;++S)if((S>>deficit)&1u&&std::popcount((unsigned)S)>=2){
        int ss=std::popcount((unsigned)S);std::array<double,N>lb{};
        for(int j=0;j<N;++j)if((S>>j)&1u)lb[j]=sat_lower(k,ss,h[j]-q[j]);
        auto vertices=dual_vertices(P,lb);if(vertices.empty())continue;
        std::vector<double>vals;vals.reserve(vertices.size());
        for(auto const&a:vertices){
            double con=0;std::array<double,M>p{};
            for(int j=0;j<N;++j)if((S>>j)&1u){
                con+=a[j]*(q[j]+1);
                for(int c=0;c<M;++c)if((A.row[j]>>c)&1u)p[c]+=a[j];
            }
            vals.push_back(con-X.max_dot(p));
        }
        double measure=0;
        if(!strict_line_safe(vals,measure)){R.coal=*std::min_element(vals.begin(),vals.end());R.zeroS=S;return R;}
        worst=std::min(worst,measure);
    }
    R.ok=true;R.sg=0;R.coal=worst;return R;
}

static double singleton_margin(const Matrix&A,const std::array<int,N>&h,int k,int deficit,const DualLine&P,CellLP&X){
    std::array<double,N>lb{};auto DV=dual_vertices(P,lb);if(DV.empty())return -1e100;double best=1e100,mx=-1e100;
    for(auto const&a:DV){std::array<double,M>p{};for(int c=0;c<M;++c)p[c]=a[deficit]*((A.row[deficit]>>c)&1u);double z=(double(k)/8.0)*(1-a[deficit])+a[deficit]*h[deficit]-X.max_dot(p);best=std::min(best,z);mx=std::max(mx,z);}
    if(best>=-1e-8&&DV.size()==2&&mx>1e-8&&best<=1e-8)return mx;
    return best;
}
static double adaptive_sum_margin(const Matrix&A,const std::array<int,N>&h,int k,int Emask,const DualLine&P,CellLP&X){
    std::array<double,N>lb{};auto DV=dual_vertices(P,lb);if(DV.empty())return -1e100;double best=1e100,mx=-1e100;int es=std::popcount((unsigned)Emask);
    for(auto const&a:DV){std::array<double,M>p{};double con=(double(k)/8.0)*es;for(int i=0;i<N;++i)if((Emask>>i)&1u){con-=double(k)/8.0*a[i];con+=a[i]*h[i];for(int c=0;c<M;++c)if((A.row[i]>>c)&1u)p[c]+=a[i];}double z=con-X.max_dot(p);best=std::min(best,z);mx=std::max(mx,z);}
    if(best>=-1e-8&&DV.size()==2&&mx>1e-8&&best<=1e-8)return mx;
    return best;
}

using Out = n8fmt::CertificateRecord;

int main(int argc,char**argv){
    if(argc!=4){std::cerr<<"usage cps_search unresolved.bin certs.bin failures.bin\n";return 2;}
    std::ifstream f(argv[1],std::ios::binary);if(!f)throw std::runtime_error(std::string("cannot open ")+argv[1]);
    u64 magic=0,count=0;f.read(reinterpret_cast<char*>(&magic),8);f.read(reinterpret_cast<char*>(&count),8);
    if(!f||magic!=n8fmt::hard_magic)throw std::runtime_error("bad or truncated hard-cell header");
    const auto expected=std::uintmax_t(16)+std::uintmax_t(sizeof(Record))*count;
    if(std::filesystem::file_size(argv[1])!=expected)throw std::runtime_error("hard-cell length mismatch");
    std::vector<Record>recs(static_cast<std::size_t>(count));
    if(!recs.empty()){f.read(reinterpret_cast<char*>(recs.data()),static_cast<std::streamsize>(recs.size()*sizeof(Record)));if(!f)throw std::runtime_error("truncated hard-cell payload");}
    for(auto const&r:recs)if(r.reserved!=0)throw std::runtime_error("nonzero hard-record reserved byte");
    std::vector<Out>ok,fail;size_t ix=0;auto t0=std::chrono::steady_clock::now();
    std::array<unsigned long long,3>types{};
    for(auto const&r:recs){++ix;Matrix A=decode_key(r.key);auto h=unpack(r.h);auto P=dual_line(A);CellLP X(A,r.k,h);Out best{};best.r=r;bool found=false;
        int E=0;std::array<double,N>coalMargin{};coalMargin.fill(-1e100);
        for(int i=0;i<N;++i)if((r.Bmask>>i)&1u){
            double bestcoal=-1e100;int bestcm=0;
            for(int cm=0;cm<(1<<M);++cm)if(std::popcount((unsigned)cm)==r.k){
                auto z=check_committee(A,h,r.k,cm,i,P,X,false);
                if(z.ok&&z.coal>bestcoal){bestcoal=z.coal;bestcm=cm;}
            }
            if(bestcoal>1e-8){E|=1<<i;best.allcm[i]=bestcm;coalMargin[i]=bestcoal;}
        }
        // A fixed deficit voter, selected after seeing the cell but with a uniform certificate.
        for(int i=0;i<N&&!found;++i)if((E>>i)&1u){
            double sg=singleton_margin(A,h,r.k,i,P,X);
            if(sg>1e-8){best.committee=best.allcm[i];best.deficit=i;best.type=0;best.sg=sg;best.coal=coalMargin[i];found=true;++types[0];}
        }
        // Adaptive deficit choice: sum of singleton margins is uniformly positive.
        if(!found&&E){
            double av=adaptive_sum_margin(A,h,r.k,E,P,X);
            if(av>1e-8){best.type=1;best.deficit=-1;best.sg=av;double wc=1e100;for(int i=0;i<N;++i)if((E>>i)&1u)wc=std::min(wc,coalMargin[i]);best.coal=wc;best.committee=E;found=true;++types[1];}
        }
        if(found)ok.push_back(best);else{best.committee=E;fail.push_back(best);++types[2];}
        if(ix%100==0){double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::cerr<<"\r"<<ix<<"/"<<count<<" ok="<<ok.size()<<" fail="<<fail.size()<<" "<<ix/sec<<"/s"<<std::flush;}
    }
    std::cerr<<"\n";
    auto save=[&](const char*path,const std::vector<Out>&v){std::ofstream o(path,std::ios::binary|std::ios::trunc);if(!o)throw std::runtime_error(std::string("cannot create ")+path);u64 n=v.size();o.write(reinterpret_cast<char*>(&n),8);if(!v.empty())o.write(reinterpret_cast<const char*>(v.data()),static_cast<std::streamsize>(v.size()*sizeof(Out)));if(!o)throw std::runtime_error(std::string("write failed for ")+path);};
    save(argv[2],ok);save(argv[3],fail);
    std::cout<<"records="<<count<<" fixed="<<types[0]<<" adaptive="<<types[1]<<" failures="<<types[2]<<"\n";
}
