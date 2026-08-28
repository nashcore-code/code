#include "n8_binary_format.hpp"
#include <algorithm>
#include <array>
#include <bit>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <gmpxx.h>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {
constexpr int N = 8;
constexpr int M = 6;
using u8 = std::uint8_t;
using i8 = std::int8_t;
using u16 = std::uint16_t;
using u32 = std::uint32_t;
using u64 = std::uint64_t;
using Rat = mpq_class;
constexpr u64 HARD_MAGIC = n8fmt::hard_magic;
constexpr char CERT_MAGIC[8] = {'M','6','C','E','R','T','0','1'};

struct Matrix {
    std::array<u8, M> col{};
    std::array<u8, N> row{};
};

Matrix decode_key(u64 key) {
    Matrix A;
    const u64 mask = (u64(1) << M) - 1;
    for (int i = N - 1; i >= 0; --i) {
        A.row[i] = static_cast<u8>(key & mask);
        key >>= M;
    }
    for (int c = 0; c < M; ++c) {
        for (int i = 0; i < N; ++i) {
            if ((A.row[i] >> c) & 1u) A.col[c] |= static_cast<u8>(1u << i);
        }
    }
    return A;
}

struct HardKey {
    u64 key{};
    u32 h{};
    u8 k{};
    u8 bmask{};
    auto tie() const { return std::tuple{key,h,k,bmask}; }
    bool operator<(const HardKey& o) const { return tie() < o.tie(); }
    bool operator==(const HardKey& o) const { return tie() == o.tie(); }
};

struct Cert {
    HardKey r;
    u8 type{};
    i8 deficit{};
    u16 committee{};
    u8 emask{};
    std::array<u16, N> allcm{};
};

template<class T>
void read_exact(std::istream& in, T& x) {
    in.read(reinterpret_cast<char*>(&x), sizeof(T));
    if (!in) throw std::runtime_error("unexpected end of file");
}

std::vector<HardKey> read_hard(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open hard file");
    u64 magic = 0, count = 0;
    read_exact(in, magic); read_exact(in, count);
    if (magic != HARD_MAGIC) throw std::runtime_error("bad hard-file magic");
    if (count > static_cast<u64>(std::numeric_limits<std::size_t>::max() / sizeof(n8fmt::HardRecord))) {
        throw std::runtime_error("hard-record count does not fit in memory");
    }
    std::vector<HardKey> out;
    out.reserve(static_cast<std::size_t>(count));
    for (u64 t = 0; t < count; ++t) {
        n8fmt::HardRecord raw{};
        read_exact(in, raw);
        if (raw.reserved != 0) throw std::runtime_error("nonzero hard-record reserved byte");
        out.push_back(HardKey{raw.key, raw.h, raw.k, raw.Bmask});
    }
    char extra;
    if (in.read(&extra,1)) throw std::runtime_error("trailing bytes in hard file");
    return out;
}

std::vector<Cert> read_certs(const std::string& path) {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open certificate file");
    char magic[8]; in.read(magic,8);
    if (!in || std::memcmp(magic,CERT_MAGIC,8)!=0) throw std::runtime_error("bad certificate magic");
    u64 count=0; read_exact(in,count);
    constexpr u64 CERT_RECORD_SIZE = 36;
    if (count > static_cast<u64>(std::numeric_limits<std::size_t>::max() / CERT_RECORD_SIZE)) {
        throw std::runtime_error("certificate count does not fit in memory");
    }
    std::vector<Cert> out; out.reserve(static_cast<std::size_t>(count));
    for (u64 t=0;t<count;++t) {
        Cert c; u8 reserved=0;
        read_exact(in,c.r.key); read_exact(in,c.r.h); read_exact(in,c.r.k); read_exact(in,c.r.bmask);
        read_exact(in,c.type); read_exact(in,c.deficit); read_exact(in,c.committee);
        read_exact(in,c.emask); read_exact(in,reserved);
        for (auto& x:c.allcm) read_exact(in,x);
        if (reserved!=0) throw std::runtime_error("nonzero reserved byte");
        if (c.type > 1) throw std::runtime_error("unknown certificate type");
        out.push_back(c);
    }
    char extra;
    if (in.read(&extra,1)) throw std::runtime_error("trailing bytes in certificate file");
    return out;
}

int hcoord(u32 h,int i) { return static_cast<int>((h>>(3*i))&7u); }
std::array<int,N> unpack(u32 h) {
    std::array<int,N> a{};
    for(int i=0;i<N;++i) a[i]=hcoord(h,i);
    return a;
}
std::array<int,N> utility(const Matrix&A,u16 cm) {
    std::array<int,N> q{};
    for(int i=0;i<N;++i) q[i]=std::popcount(static_cast<unsigned>(A.row[i]&cm));
    return q;
}
void verify_puncture(const Matrix&A,const std::array<int,N>&h,int k,u16 cm,int i,const HardKey&r) {
    if(std::popcount(static_cast<unsigned>(cm))!=k) {
        throw std::runtime_error("committee-size failure at key "+std::to_string(r.key));
    }
    auto q=utility(A,cm);
    if(q[i]!=h[i]-1) throw std::runtime_error("deficit failure at key "+std::to_string(r.key));
    for(int j=0;j<N;++j) if(j!=i&&q[j]<h[j]) {
        throw std::runtime_error("puncture-cover failure at key "+std::to_string(r.key));
    }
}

Rat saturation_lower(int k,int s,int d) {
    Rat best(1,s);
    for(int K=k;K<k+8;++K) {
        int den=8*((K*s)/8+d);
        if(den>0) {
            Rat z(K,den);
            if(z<best) best=z;
        }
    }
    return best;
}

struct ExactLP {
    int m,n;
    std::vector<int> B,Nv;
    std::vector<std::vector<Rat>> D;
    ExactLP(const std::vector<std::vector<long long>>&A,
            const std::vector<long long>&b,
            const std::vector<Rat>&c)
      :m(static_cast<int>(b.size())),n(static_cast<int>(c.size())),
       B(m),Nv(n+1),D(m+2,std::vector<Rat>(n+2)) {
        for(int i=0;i<m;++i)for(int j=0;j<n;++j)D[i][j]=Rat(static_cast<long>(A[i][j]));
        for(int i=0;i<m;++i){B[i]=n+i;D[i][n]=-1;D[i][n+1]=Rat(static_cast<long>(b[i]));}
        for(int j=0;j<n;++j){Nv[j]=j;D[m][j]=-c[j];}
        Nv[n]=-1;D[m+1][n]=1;
    }
    void pivot(int r,int s) {
        Rat inv=Rat(1)/D[r][s];
        for(int i=0;i<m+2;++i)if(i!=r)for(int j=0;j<n+2;++j)if(j!=s)
            D[i][j]-=D[r][j]*D[i][s]*inv;
        for(int j=0;j<n+2;++j)if(j!=s)D[r][j]*=inv;
        for(int i=0;i<m+2;++i)if(i!=r)D[i][s]*=-inv;
        D[r][s]=inv;std::swap(B[r],Nv[s]);
    }
    bool simplex(int phase) {
        int x=(phase==1?m+1:m);
        for(;;) {
            int s=-1;
            for(int j=0;j<=n;++j) {
                if(phase==2&&Nv[j]==-1)continue;
                if(s==-1||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&Nv[j]<Nv[s]))s=j;
            }
            if(D[x][s]>=0)return true;
            int r=-1;
            for(int i=0;i<m;++i) {
                if(D[i][s]<=0)continue;
                if(r==-1)r=i;
                else {
                    Rat l=D[i][n+1]/D[i][s],z=D[r][n+1]/D[r][s];
                    if(l<z||(l==z&&B[i]<B[r]))r=i;
                }
            }
            if(r==-1)return false;
            pivot(r,s);
        }
    }
    Rat solve() {
        int r=0;for(int i=1;i<m;++i)if(D[i][n+1]<D[r][n+1])r=i;
        if(D[r][n+1]<0) {
            pivot(r,n);
            if(!simplex(1)||D[m+1][n+1]<0)throw std::runtime_error("certificate cell infeasible");
            auto it=std::find(B.begin(),B.end(),-1);
            if(it!=B.end()) {
                r=static_cast<int>(it-B.begin());int s=-1;
                for(int j=0;j<=n;++j)if(D[r][j]!=0&&(s==-1||D[r][j]<D[r][s]||(D[r][j]==D[r][s]&&Nv[j]<Nv[s])))s=j;
                if(s!=-1)pivot(r,s);
            }
        }
        if(!simplex(2))throw std::runtime_error("unbounded cell objective");
        return D[m][n+1];
    }
};

struct Cell {
    std::vector<std::vector<long long>> C;
    std::vector<long long> b;
    int k;
    std::map<std::array<Rat,M>,Rat> cache;
    Cell(const Matrix&A,int kk,const std::array<int,N>&h):k(kk) {
        const int V=M-1;
        auto add=[&](std::vector<long long>a,long long rhs){C.push_back(std::move(a));b.push_back(rhs);};
        for(int c=0;c<V;++c){std::vector<long long>a(V);a[c]=1;add(a,1);}
        {std::vector<long long>a(V,1);add(a,k);}       // x_M >= 0
        {std::vector<long long>a(V,-1);add(a,1-k);}  // x_M <= 1
        for(int i=0;i<N;++i) {
            int last=(A.row[i]>>(M-1))&1u;
            long long con=last*k;
            std::vector<long long>lo(V),up(V);
            for(int c=0;c<V;++c) {
                long long q=static_cast<int>((A.row[i]>>c)&1u)-last;
                lo[c]=-q;up[c]=q;
            }
            add(lo,con-h[i]);
            add(up,h[i]+1-con);
        }
    }
    Rat maxdot(const std::array<Rat,M>&p) {
        auto it=cache.find(p);if(it!=cache.end())return it->second;
        const int V=M-1;
        std::vector<Rat> obj(V);
        for(int c=0;c<V;++c)obj[c]=p[c]-p[M-1];
        ExactLP lp(C,b,obj);
        Rat ans=p[M-1]*Rat(k)+lp.solve();
        cache.emplace(p,ans);
        return ans;
    }
};

std::string rat_string(const Rat&x) {
    std::ostringstream out;out<<x.get_num()<<'/'<<x.get_den();return out.str();
}
std::string alpha_key(const std::array<Rat,N>&a) {
    std::ostringstream out;
    for(const auto&x:a)out<<x.get_num()<<'/'<<x.get_den()<<';';
    return out.str();
}

bool solve_dual_vertex(const Matrix&A,const std::array<Rat,N>&lb,int p,int q,std::array<Rat,N>&sol) {
    std::array<std::array<Rat,N+1>,N> R{};
    for(int c=0;c<M;++c) {
        for(int i=0;i<N;++i)R[c][i]=Rat((A.row[i]>>c)&1u);
        R[c][N]=1;
    }
    R[M][p]=1;R[M][N]=lb[p];
    R[M+1][q]=1;R[M+1][N]=lb[q];
    int rank=0;
    std::array<int,N> pivot_col{};pivot_col.fill(-1);
    for(int col=0;col<N&&rank<N;++col) {
        int r=rank;while(r<N&&R[r][col]==0)++r;
        if(r==N)continue;
        std::swap(R[r],R[rank]);
        Rat z=R[rank][col];
        for(int j=col;j<=N;++j)R[rank][j]/=z;
        for(int rr=0;rr<N;++rr)if(rr!=rank&&R[rr][col]!=0) {
            Rat f=R[rr][col];
            for(int j=col;j<=N;++j)R[rr][j]-=f*R[rank][j];
        }
        pivot_col[rank]=col;++rank;
    }
    if(rank<N)return false;
    sol.fill(0);
    for(int r=0;r<N;++r)sol[pivot_col[r]]=R[r][N];
    for(auto&x:sol)x.canonicalize();
    for(int i=0;i<N;++i)if(sol[i]<lb[i])return false;
    for(int c=0;c<M;++c) {
        Rat s=0;for(int i=0;i<N;++i)if((A.row[i]>>c)&1u)s+=sol[i];
        s.canonicalize();
        if(s!=1)return false;
    }
    return true;
}

std::vector<std::array<Rat,N>> dual_vertices(const Matrix&A,const std::array<Rat,N>&lb) {
    std::vector<std::array<Rat,N>> out;
    std::set<std::string> seen;
    for(int p=0;p<N;++p)for(int q=p+1;q<N;++q) {
        std::array<Rat,N>a{};
        if(solve_dual_vertex(A,lb,p,q,a)) {
            auto key=alpha_key(a);
            if(seen.insert(key).second)out.push_back(std::move(a));
        }
    }
    return out;
}

Rat minimum_singleton(const Matrix&A,const std::array<int,N>&h,int k,int i,Cell&cell) {
    std::array<Rat,N>lb{};
    auto verts=dual_vertices(A,lb);
    if(verts.empty())throw std::runtime_error("positive-dual closure has no vertex");
    bool have=false;Rat best;
    for(const auto&a:verts) {
        std::array<Rat,M>p{};
        for(int c=0;c<M;++c)if((A.row[i]>>c)&1u)p[c]=a[i];
        Rat v=Rat(k,8)*(Rat(1)-a[i])+a[i]*Rat(h[i])-cell.maxdot(p);
        if(!have||v<best){best=v;have=true;}
    }
    return best;
}

Rat minimum_adaptive_sum(const Matrix&A,const std::array<int,N>&h,int k,int E,Cell&cell) {
    std::array<Rat,N>lb{};
    auto verts=dual_vertices(A,lb);
    if(verts.empty())throw std::runtime_error("positive-dual closure has no vertex");
    bool have=false;Rat best;const int es=std::popcount(static_cast<unsigned>(E));
    for(const auto&a:verts) {
        Rat sum=0,hh=0;std::array<Rat,M>p{};
        for(int i=0;i<N;++i)if((E>>i)&1u) {
            sum+=a[i];hh+=a[i]*Rat(h[i]);
            for(int c=0;c<M;++c)if((A.row[i]>>c)&1u)p[c]+=a[i];
        }
        Rat v=Rat(k,8)*(Rat(es)-sum)+hh-cell.maxdot(p);
        if(!have||v<best){best=v;have=true;}
    }
    return best;
}

// Returns false exactly when the closed saturation-restricted dual domain is empty.
bool minimum_coalition(const Matrix&A,const std::array<int,N>&h,int k,u16 cm,int S,Cell&cell,Rat&best) {
    auto q=utility(A,cm);
    const int s=std::popcount(static_cast<unsigned>(S));
    std::array<Rat,N>lb{};
    for(int j=0;j<N;++j)if((S>>j)&1u)lb[j]=saturation_lower(k,s,h[j]-q[j]);
    auto verts=dual_vertices(A,lb);
    if(verts.empty())return false;
    bool have=false;
    for(const auto&a:verts) {
        Rat con=0;std::array<Rat,M>p{};
        for(int j=0;j<N;++j)if((S>>j)&1u) {
            con+=a[j]*Rat(q[j]+1);
            for(int c=0;c<M;++c)if((A.row[j]>>c)&1u)p[c]+=a[j];
        }
        Rat v=con-cell.maxdot(p);
        if(!have||v<best){best=v;have=true;}
    }
    return true;
}

} // namespace

int main(int argc,char**argv) {
    try {
        if(argc!=3) {
            std::cerr<<"usage: m6_exact_certificate_checker hard.bin certificates.bin\n";
            return 2;
        }
        auto hard=read_hard(argv[1]);
        auto certs=read_certs(argv[2]);
        if(hard.size()!=certs.size())throw std::runtime_error("hard/certificate count mismatch");
        std::set<HardKey> hardset(hard.begin(),hard.end()),certset;
        if(hardset.size()!=hard.size())throw std::runtime_error("duplicate hard record");
        for(const auto&c:certs) {
            if(!certset.insert(c.r).second)throw std::runtime_error("duplicate certificate record");
        }
        if(hardset!=certset)throw std::runtime_error("hard/certificate record-set mismatch");

        std::map<HardKey,const Cert*> bykey;
        for(const auto&c:certs)bykey[c.r]=&c;
        std::size_t fixed=0,adaptive=0,empty_domains=0,price_domains=0;
        bool have_single=false,have_price=false;Rat min_single,min_price;
        for(std::size_t ix=0;ix<hard.size();++ix) {
            const HardKey&r=hard[ix];const Cert&c=*bykey.at(r);
            Matrix A=decode_key(r.key);auto h=unpack(r.h);Cell cell(A,r.k,h);
            // Force an exact feasibility check of the stored closed cell.
            std::array<Rat,M>zero{}; (void)cell.maxdot(zero);
            auto check_coalitions=[&](int i,u16 cm) {
                verify_puncture(A,h,r.k,cm,i,r);
                for(int S=1;S<(1<<N);++S) {
                    if(!((S>>i)&1u)||std::popcount(static_cast<unsigned>(S))<2)continue;
                    Rat v;
                    if(!minimum_coalition(A,h,r.k,cm,S,cell,v)) {++empty_domains;continue;}
                    ++price_domains;
                    if(!(v>0)) {
                        throw std::runtime_error("nonpositive coalition margin at key "+std::to_string(r.key)+
                                                 " S="+std::to_string(S)+" margin="+rat_string(v));
                    }
                    if(!have_price||v<min_price){min_price=v;have_price=true;}
                }
            };
            if(c.type==0) {
                ++fixed;
                if(c.deficit<0||c.deficit>=N)throw std::runtime_error("bad fixed deficit");
                if(!((r.bmask>>c.deficit)&1u))throw std::runtime_error("fixed deficit not usable");
                check_coalitions(c.deficit,c.committee);
                Rat v=minimum_singleton(A,h,r.k,c.deficit,cell);
                if(!(v>0))throw std::runtime_error("nonpositive singleton margin at key "+std::to_string(r.key)+" margin="+rat_string(v));
                if(!have_single||v<min_single){min_single=v;have_single=true;}
            } else if(c.type==1) {
                ++adaptive;
                if(c.emask==0||(c.emask&~r.bmask))throw std::runtime_error("invalid adaptive mask");
                for(int i=0;i<N;++i)if((c.emask>>i)&1u)check_coalitions(i,c.allcm[i]);
                Rat v=minimum_adaptive_sum(A,h,r.k,c.emask,cell);
                if(!(v>0))throw std::runtime_error("nonpositive adaptive-sum margin at key "+std::to_string(r.key)+" margin="+rat_string(v));
                if(!have_single||v<min_single){min_single=v;have_single=true;}
            } else throw std::runtime_error("unknown certificate type");
            if((ix+1)%20==0)std::cerr<<"\r"<<(ix+1)<<'/'<<hard.size()<<std::flush;
        }
        std::cerr<<"\n";
        std::cout<<"PASS certs="<<certs.size()<<" fixed="<<fixed<<" adaptive="<<adaptive
                 <<" empty_saturation_domains="<<empty_domains<<" exact_price_domains="<<price_domains
                 <<" min_singleton_or_sum="<<(have_single?rat_string(min_single):"NA")
                 <<" min_exact_price="<<(have_price?rat_string(min_price):"NA")<<"\n";
        return 0;
    } catch(const std::exception&e) {
        std::cerr<<"FAIL: "<<e.what()<<"\n";
        return 1;
    }
}
