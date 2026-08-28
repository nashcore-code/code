#include "n8_binary_format.hpp"
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <functional>
#include <filesystem>
#include <limits>
#include <stdexcept>
#include <iostream>
#include <string>
#include <thread>
#include <unordered_set>
#include <utility>
#include <vector>

using u8 = std::uint8_t;
using u64 = std::uint64_t;
using i128 = __int128_t;
constexpr int MAXN = 8;
constexpr int MAXM = 8;
constexpr int PRIME = 1000003; // exceeds Hadamard bound here

struct Matrix {
    int n = 0, m = 0;
    std::array<u8, MAXM> col{};
};

static void save_keys(const std::string& path, const std::vector<u64>& v) {
    std::ofstream f(path, std::ios::binary | std::ios::trunc);
    if (!f) throw std::runtime_error("cannot create " + path);
    const u64 n = static_cast<u64>(v.size());
    f.write(reinterpret_cast<const char*>(&n), sizeof(n));
    if (!v.empty()) {
        f.write(reinterpret_cast<const char*>(v.data()),
                static_cast<std::streamsize>(v.size() * sizeof(u64)));
    }
    if (!f) throw std::runtime_error("write failed for " + path);
}
static std::vector<u64> load_keys(const std::string& path) {
    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("cannot open " + path);
    u64 n = 0;
    f.read(reinterpret_cast<char*>(&n), sizeof(n));
    if (!f) throw std::runtime_error("truncated key-list header: " + path);
    if (n > static_cast<u64>(std::numeric_limits<std::size_t>::max() / sizeof(u64)))
        throw std::runtime_error("key-list count too large: " + path);
    const auto expected = std::uintmax_t(8) + std::uintmax_t(8) * n;
    if (std::filesystem::file_size(path) != expected)
        throw std::runtime_error("key-list length mismatch: " + path);
    std::vector<u64> v(static_cast<std::size_t>(n));
    if (!v.empty()) {
        f.read(reinterpret_cast<char*>(v.data()),
               static_cast<std::streamsize>(v.size() * sizeof(u64)));
        if (!f) throw std::runtime_error("truncated key-list payload: " + path);
    }
    return v;
}

static Matrix decode_key(u64 key, int n, int m) {
    Matrix A; A.n=n; A.m=m;
    std::array<u8,MAXN> rows{};
    const u64 mask=(u64(1)<<m)-1;
    for(int i=n-1;i>=0;--i){rows[i]=u8(key&mask); key>>=m;}
    for(int c=0;c<m;++c) for(int i=0;i<n;++i)
        if((rows[i]>>c)&1u) A.col[c]|=u8(1u<<i);
    return A;
}

static int rank_mod(const Matrix& A) {
    int a[MAXN][MAXM]{};
    for(int i=0;i<A.n;++i) for(int c=0;c<A.m;++c) a[i][c]=(A.col[c]>>i)&1u;
    auto modpow=[](long long x,int e){long long y=1;while(e){if(e&1)y=y*x%PRIME;x=x*x%PRIME;e>>=1;}return int(y);};
    int r=0;
    for(int c=0;c<A.m && r<A.n;++c){
        int p=r; while(p<A.n && a[p][c]==0) ++p;
        if(p==A.n) continue;
        for(int j=c;j<A.m;++j) std::swap(a[p][j],a[r][j]);
        int inv=modpow(a[r][c],PRIME-2);
        for(int j=c;j<A.m;++j) a[r][j]=int((long long)a[r][j]*inv%PRIME);
        for(int i=0;i<A.n;++i) if(i!=r && a[i][c]){
            int f=a[i][c];
            for(int j=c;j<A.m;++j){
                int x=a[i][j]-int((long long)f*a[r][j]%PRIME);
                if(x<0)x+=PRIME; a[i][j]=x;
            }
        }
        ++r;
    }
    return r;
}

// Exact canonical form under voter and candidate relabeling. Voter labels are
// removed by sorting row patterns. Stable bipartite color refinement restricts
// the candidate permutations that remain to be inspected.
static u64 canonical_key(const Matrix& A) {
    const int n=A.n,m=A.m,V=n+m;
    std::array<int,16> color{}, next{};
    for(int i=0;i<n;++i) color[i]=0;
    for(int c=0;c<m;++c) color[n+c]=1;
    for(;;){
        int C=*std::max_element(color.begin(),color.begin()+V)+1;
        struct Sig{int v,side,old;std::array<u8,16> cnt{};};
        std::vector<Sig> ss; ss.reserve(V);
        for(int v=0;v<V;++v){
            Sig s{}; s.v=v;s.side=(v<n?0:1);s.old=color[v];
            if(v<n){for(int c=0;c<m;++c)if((A.col[c]>>v)&1u)++s.cnt[color[n+c]];}
            else {int c=v-n;for(int i=0;i<n;++i)if((A.col[c]>>i)&1u)++s.cnt[color[i]];}
            ss.push_back(s);
        }
        std::sort(ss.begin(),ss.end(),[C](const Sig&a,const Sig&b){
            if(a.side!=b.side)return a.side<b.side;if(a.old!=b.old)return a.old<b.old;
            for(int k=0;k<C;++k)if(a.cnt[k]!=b.cnt[k])return a.cnt[k]<b.cnt[k];
            return a.v<b.v;
        });
        int id=-1; Sig* prev=nullptr;
        for(auto& s:ss){
            bool diff=!prev||s.side!=prev->side||s.old!=prev->old;
            if(!diff)for(int k=0;k<C;++k)if(s.cnt[k]!=prev->cnt[k]){diff=true;break;}
            if(diff)++id; next[s.v]=id; prev=&s;
        }
        bool same=true;for(int v=0;v<V;++v)if(next[v]!=color[v]){same=false;break;}
        color=next;if(same)break;
    }
    std::vector<std::pair<int,std::vector<int>>> cls;
    for(int c=0;c<m;++c){int cc=color[n+c];auto it=std::find_if(cls.begin(),cls.end(),[cc](auto const&p){return p.first==cc;});if(it==cls.end())cls.push_back({cc,{c}});else it->second.push_back(c);}
    std::sort(cls.begin(),cls.end(),[](auto const&a,auto const&b){return a.first<b.first;});
    std::array<int,MAXM> old_at{};u64 best=~u64(0);
    std::function<void(int,int)> rec=[&](int k,int off){
        if(k==(int)cls.size()){
            std::array<u8,MAXN> rows{};
            for(int i=0;i<n;++i){u8 r=0;for(int t=0;t<m;++t)if((A.col[old_at[t]]>>i)&1u)r|=u8(1u<<t);rows[i]=r;}
            std::sort(rows.begin(),rows.begin()+n);u64 key=0;for(int i=0;i<n;++i)key=(key<<m)|rows[i];best=std::min(best,key);return;
        }
        auto v=cls[k].second;std::sort(v.begin(),v.end());do{for(int j=0;j<(int)v.size();++j)old_at[off+j]=v[j];rec(k+1,off+(int)v.size());}while(std::next_permutation(v.begin(),v.end()));
    };rec(0,0);return best;
}

static bool antichain_and_fullrank(const Matrix& A){
    const u8 all=u8((1u<<A.n)-1u);
    for(int c=0;c<A.m;++c)if(A.col[c]==0||A.col[c]==all)return false;
    for(int a=0;a<A.m;++a)for(int b=a+1;b<A.m;++b){u8 x=A.col[a],y=A.col[b];if((x&y)==x||(x&y)==y)return false;}
    return rank_mod(A)==A.m;
}

static std::vector<u64> direct_rows(int n,int m,int threads){
    std::vector<std::unordered_set<u64>> local(threads);
    auto worker=[&](int tid){
        auto& out=local[tid];out.reserve(200000);std::array<u8,MAXN> rows{};
        std::function<void(int,int)> rec=[&](int pos,int lo){
            if(pos==n){Matrix A;A.n=n;A.m=m;for(int c=0;c<m;++c)for(int i=0;i<n;++i)if((rows[i]>>c)&1u)A.col[c]|=u8(1u<<i);if(antichain_and_fullrank(A))out.insert(canonical_key(A));return;}
            for(int r=lo;r<(1<<m);++r){if(pos==0&&r%threads!=tid)continue;rows[pos]=u8(r);rec(pos+1,r);}
        };rec(0,0);
    };
    std::vector<std::thread> ts;for(int t=0;t<threads;++t)ts.emplace_back(worker,t);for(auto&t:ts)t.join();
    std::vector<u64> all;size_t total=0;for(auto&s:local)total+=s.size();all.reserve(total);for(auto&s:local)for(u64 k:s)all.push_back(k);std::sort(all.begin(),all.end());all.erase(std::unique(all.begin(),all.end()),all.end());return all;
}

static std::vector<u64> extend_one(const std::vector<u64>& parents,int n,int m,int threads){
    const int cm=m+1;std::vector<std::vector<u64>> local(threads);std::atomic<size_t> done=0;auto t0=std::chrono::steady_clock::now();
    auto worker=[&](int tid){auto&out=local[tid];out.reserve((parents.size()/threads+1)*40);for(size_t ix=tid;ix<parents.size();ix+=threads){Matrix A=decode_key(parents[ix],n,m);for(int s=1;s<(1<<n)-1;++s){bool ok=true;for(int c=0;c<m;++c){u8 x=A.col[c],y=u8(s);if((x&y)==x||(x&y)==y){ok=false;break;}}if(!ok)continue;Matrix B=A;B.m=cm;B.col[m]=u8(s);if(rank_mod(B)!=cm)continue;out.push_back(canonical_key(B));}size_t d=++done;if(tid==0&&d%10000==0){double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();std::cerr<<"\r"<<d<<"/"<<parents.size()<<" parents ("<<d/sec<<"/s)"<<std::flush;}}};
    std::vector<std::thread>ts;for(int t=0;t<threads;++t)ts.emplace_back(worker,t);for(auto&t:ts)t.join();std::cerr<<"\n";size_t total=0;for(auto&v:local)total+=v.size();std::vector<u64>all;all.reserve(total);for(auto&v:local){all.insert(all.end(),v.begin(),v.end());std::vector<u64>().swap(v);}std::sort(all.begin(),all.end());all.erase(std::unique(all.begin(),all.end()),all.end());return all;
}

static long long det_small(std::array<std::array<long long,MAXM>,MAXM> a,int k){
    if(k==0)return 1; i128 prev=1;int sign=1;
    for(int c=0;c<k-1;++c){int p=c;while(p<k&&a[p][c]==0)++p;if(p==k)return 0;if(p!=c){std::swap(a[p],a[c]);sign=-sign;}i128 pivot=a[c][c];for(int i=c+1;i<k;++i)for(int j=c+1;j<k;++j){i128 num=i128(a[i][j])*pivot-i128(a[i][c])*a[c][j];num/=prev;a[i][j]=(long long)num;}prev=pivot;if(prev==0)return 0;}
    return sign*a[k-1][k-1];
}

// A^T alpha=1 has alpha>0 iff 1 lies in the interior of the cone generated
// by the row vectors. Check all extreme rays of the dual cone exactly.
static bool positive_dual_exact(const Matrix& A){
    const int n=A.n,m=A.m,need=m-1;std::array<std::array<long long,MAXM>,MAXN> row{};
    for(int i=0;i<n;++i)for(int c=0;c<m;++c)row[i][c]=(A.col[c]>>i)&1u;
    for(int mask=0;mask<(1<<n);++mask)if(__builtin_popcount((unsigned)mask)==need){
        int ids[MAXN],q=0;for(int i=0;i<n;++i)if((mask>>i)&1)ids[q++]=i;
        long long y[MAXM]{};bool nonzero=false;
        for(int omit=0;omit<m;++omit){std::array<std::array<long long,MAXM>,MAXM>B{};for(int r=0;r<need;++r){int cc=0;for(int c=0;c<m;++c)if(c!=omit)B[r][cc++]=row[ids[r]][c];}long long d=det_small(B,need);if(omit&1)d=-d;y[omit]=d;nonzero|=(d!=0);}
        if(!nonzero)continue;bool ge=true,le=true;for(int i=0;i<n;++i){long long dot=0;for(int c=0;c<m;++c)dot+=row[i][c]*y[c];ge&=(dot>=0);le&=(dot<=0);}long long sum=0;for(int c=0;c<m;++c)sum+=y[c];if(ge&&sum<=0)return false;if(le&&sum>=0)return false;
    }
    return true;
}

static std::vector<u64> filter_positive(const std::vector<u64>&v,int n,int m,int threads){std::vector<std::vector<u64>>local(threads);auto work=[&](int t){for(size_t i=t;i<v.size();i+=threads){Matrix A=decode_key(v[i],n,m);if(positive_dual_exact(A))local[t].push_back(v[i]);}};std::vector<std::thread>ts;for(int t=0;t<threads;++t)ts.emplace_back(work,t);for(auto&t:ts)t.join();std::vector<u64>all;size_t z=0;for(auto&o:local)z+=o.size();all.reserve(z);for(auto&o:local)all.insert(all.end(),o.begin(),o.end());std::sort(all.begin(),all.end());return all;}



static bool positive_square_exact(const Matrix& A){
    const int n=A.n,m=A.m;
    if(n!=m)return positive_dual_exact(A);
    std::array<std::array<long long,MAXM>,MAXM> M{};
    for(int c=0;c<m;++c)for(int i=0;i<n;++i)M[c][i]=(A.col[c]>>i)&1u;
    long long D=det_small(M,m);
    if(D==0)return false;
    for(int j=0;j<n;++j){
        auto B=M;for(int r=0;r<m;++r)B[r][j]=1;
        long long d=det_small(B,m);
        if((D>0&&d<=0)||(D<0&&d>=0))return false;
    }
    return true;
}

static std::vector<u64> extend_one_positive(const std::vector<u64>& parents,int n,int m,int threads){
    const int cm=m+1;
    std::vector<std::vector<u64>> local(threads);
    std::atomic<size_t> next{0},done{0};
    auto t0=std::chrono::steady_clock::now();
    auto worker=[&](int tid){
        auto& out=local[tid];
        out.reserve((parents.size()/threads+1)*20);
        for(;;){
            size_t ix=next.fetch_add(1,std::memory_order_relaxed);
            if(ix>=parents.size()) break;
            Matrix A=decode_key(parents[ix],n,m);
            for(int s=1;s<(1<<n)-1;++s){
                bool ok=true;
                for(int c=0;c<m;++c){
                    u8 x=A.col[c],y=u8(s);
                    if((x&y)==x||(x&y)==y){ok=false;break;}
                }
                if(!ok) continue;
                Matrix B=A;B.m=cm;B.col[m]=u8(s);
                // In the square stage, positive_square_exact already checks a
                // nonzero determinant, so a separate modular-rank computation
                // would be redundant.  Retain the rank test for rectangular
                // children before the positive-dual cone check.
                if(cm==n){
                    if(!positive_square_exact(B)) continue;
                }else{
                    if(rank_mod(B)!=cm) continue;
                    if(!positive_dual_exact(B)) continue;
                }
                out.push_back(canonical_key(B));
            }
            size_t d=done.fetch_add(1)+1;
            if(tid==0 && d%10000==0){
                double sec=std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count();
                std::cerr<<"\r"<<d<<"/"<<parents.size()<<" parents ("<<d/sec<<"/s), raw=";
                size_t raw=0;for(auto const&v:local)raw+=v.size();
                std::cerr<<raw<<std::flush;
            }
        }
        std::sort(out.begin(),out.end());
        out.erase(std::unique(out.begin(),out.end()),out.end());
    };
    std::vector<std::thread> ts;
    for(int t=0;t<threads;++t)ts.emplace_back(worker,t);
    for(auto&t:ts)t.join();
    std::cerr<<"\n";
    size_t total=0;for(auto&v:local)total+=v.size();
    std::vector<u64> all;all.reserve(total);
    for(auto&v:local){all.insert(all.end(),v.begin(),v.end());std::vector<u64>().swap(v);}
    std::sort(all.begin(),all.end());
    all.erase(std::unique(all.begin(),all.end()),all.end());
    return all;
}

int main(int argc,char**argv){
    if(argc<2){std::cerr<<"modes: direct n m threads out | extend n m threads in out | extendpos n m threads in out | positive n m threads in out\n";return 2;}
    std::string mode=argv[1];auto t0=std::chrono::steady_clock::now();
    auto validate=[](int n,int m,int T){
        if(n<1||n>MAXN||m<1||m>MAXM||m>n||T<1)
            throw std::invalid_argument("require 1<=m<=n<=8 and threads>=1");
    };
    if(mode=="direct"&&argc==6){int n=atoi(argv[2]),m=atoi(argv[3]),T=atoi(argv[4]);validate(n,m,T);auto v=direct_rows(n,m,T);save_keys(argv[5],v);std::cout<<"direct n="<<n<<" m="<<m<<" full_rank_antichain_orbits="<<v.size();}
    else if(mode=="extend"&&argc==7){int n=atoi(argv[2]),m=atoi(argv[3]),T=atoi(argv[4]);validate(n,m,T);if(m>=MAXM)throw std::invalid_argument("cannot extend m=8");auto p=load_keys(argv[5]);auto v=extend_one(p,n,m,T);save_keys(argv[6],v);std::cout<<"extend n="<<n<<" "<<m<<"->"<<m+1<<" orbits="<<v.size();}
    else if(mode=="extendpos"&&argc==7){int n=atoi(argv[2]),m=atoi(argv[3]),T=atoi(argv[4]);validate(n,m,T);if(m>=MAXM)throw std::invalid_argument("cannot extend m=8");auto p=load_keys(argv[5]);auto v=extend_one_positive(p,n,m,T);save_keys(argv[6],v);std::cout<<"extendpos n="<<n<<" "<<m<<"->"<<m+1<<" positive_orbits="<<v.size();}
    else if(mode=="positive"&&argc==7){int n=atoi(argv[2]),m=atoi(argv[3]),T=atoi(argv[4]);validate(n,m,T);auto v=load_keys(argv[5]);auto p=filter_positive(v,n,m,T);save_keys(argv[6],p);std::cout<<"positive n="<<n<<" m="<<m<<" total="<<v.size()<<" positive="<<p.size();}
    else {std::cerr<<"bad arguments\n";return 2;}
    std::cout<<" seconds="<<std::chrono::duration<double>(std::chrono::steady_clock::now()-t0).count()<<"\n";
}
