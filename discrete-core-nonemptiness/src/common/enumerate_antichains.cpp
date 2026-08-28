
#include <bits/stdc++.h>
using namespace std;
struct Sig{vector<int>v;bool operator<(Sig const&o)const{return v<o.v;}bool operator==(Sig const&o)const{return v==o.v;}};
class E{
 int n,mm;
public:E(int q):n(q),mm((1<<q)-1){}
 vector<int> dec(uint64_t z,int m)const{vector<int>e(m);for(int i=m-1;i>=0;--i){e[i]=z&mm;z>>=n;}return e;}
 vector<int> colors(vector<int> const&e)const{
  vector<vector<int>>inc(n);vector<vector<int>>cd(n,vector<int>(n));
  for(int a:e){int s=__builtin_popcount((unsigned)a);for(int i=0;i<n;i++)if(a>>i&1)inc[i].push_back(s);for(int i=0;i<n;i++)if(a>>i&1)for(int j=i+1;j<n;j++)if(a>>j&1)cd[i][j]++,cd[j][i]++;}
  vector<Sig>g(n);for(int i=0;i<n;i++){sort(inc[i].begin(),inc[i].end());g[i].v.push_back(inc[i].size());g[i].v.insert(g[i].v.end(),inc[i].begin(),inc[i].end());vector<int>x;for(int j=0;j<n;j++)if(i!=j)x.push_back(cd[i][j]);sort(x.begin(),x.end());g[i].v.push_back(-1);g[i].v.insert(g[i].v.end(),x.begin(),x.end());}
  vector<int>col(n);
  for(int r=0;r<n;r++){map<Sig,int>mp;for(auto&s:g)mp[s]=0;int q=0;for(auto&z:mp)z.second=q++;for(int i=0;i<n;i++)col[i]=mp[g[i]];vector<Sig>h(n);
   for(int i=0;i<n;i++){h[i].v={col[i]};vector<vector<int>>ev;for(int a:e)if(a>>i&1){vector<int>x{__builtin_popcount((unsigned)a)};for(int j=0;j<n;j++)if(a>>j&1)x.push_back(col[j]);sort(x.begin()+1,x.end());ev.push_back(x);}sort(ev.begin(),ev.end());for(auto&x:ev){h[i].v.push_back(-2);h[i].v.insert(h[i].v.end(),x.begin(),x.end());}vector<pair<int,int>>p;for(int j=0;j<n;j++)if(i!=j)p.push_back({col[j],cd[i][j]});sort(p.begin(),p.end());h[i].v.push_back(-3);for(auto z:p){h[i].v.push_back(z.first);h[i].v.push_back(z.second);}}
   if(h==g)break;g.swap(h);
  }
  map<Sig,int>mp;for(auto&s:g)mp[s]=0;int q=0;for(auto&z:mp)z.second=q++;for(int i=0;i<n;i++)col[i]=mp[g[i]];return col;
 }
 uint64_t canon(vector<int> const&e)const{
  auto col=colors(e);int nc=*max_element(col.begin(),col.end())+1;vector<vector<int>>G(nc);for(int i=0;i<n;i++)G[col[i]].push_back(i);vector<int>st(nc);int c=0;for(int j=0;j<nc;j++){st[j]=c;c+=G[j].size();}vector<int>mpv(n,-1);uint64_t best=UINT64_MAX;
  function<void(int)>rec=[&](int g){if(g==nc){vector<int>a(e.size());for(size_t q=0;q<e.size();q++){int y=0;for(int i=0;i<n;i++)if(e[q]>>i&1)y|=1<<mpv[i];a[q]=y;}sort(a.begin(),a.end());uint64_t z=0;for(int y:a)z=(z<<n)|y;best=min(best,z);return;}auto old=G[g];vector<int>lab(old.size());iota(lab.begin(),lab.end(),st[g]);do{for(size_t j=0;j<old.size();j++)mpv[old[j]]=lab[j];rec(g+1);}while(next_permutation(lab.begin(),lab.end()));};rec(0);return best;
 }
 int rank(vector<int>const&e)const{
  const long long P=1000000007LL;int m=e.size();vector<vector<long long>>a(n,vector<long long>(m));for(int i=0;i<n;i++)for(int j=0;j<m;j++)a[i][j]=(e[j]>>i)&1;
  auto pw=[&](long long x,long long k){long long r=1;while(k){if(k&1)r=r*x%P;x=x*x%P;k>>=1;}return r;};int r=0;
  for(int j=0;j<m&&r<n;j++){int p=r;while(p<n&&!a[p][j])p++;if(p==n)continue;swap(a[p],a[r]);long long inv=pw(a[r][j],P-2);for(int z=j;z<m;z++)a[r][z]=a[r][z]*inv%P;for(int i=0;i<n;i++)if(i!=r&&a[i][j]){long long f=a[i][j];for(int z=j;z<m;z++){a[i][z]=(a[i][z]-f*a[r][z])%P;if(a[i][z]<0)a[i][z]+=P;}}r++;}return r;
 }
 void run(int lo,int hi,string out)const{
  vector<unordered_set<uint64_t>>L(hi+1);for(int a=1;a<mm;a++)L[1].insert(canon({a}));cerr<<"level 1 "<<L[1].size()<<"\n";
  for(int r=1;r<hi;r++){for(auto z:L[r]){auto e=dec(z,r);for(int a=1;a<mm;a++){bool ok=1;for(int b:e)if((a&b)==a||(a&b)==b){ok=0;break;}if(ok){auto f=e;f.push_back(a);L[r+1].insert(canon(f));}}}cerr<<"level "<<r+1<<" "<<L[r+1].size()<<"\n";}
  ofstream f(out);long long tot=0;for(int r=lo;r<=hi;r++){vector<uint64_t>v(L[r].begin(),L[r].end());sort(v.begin(),v.end());long long q=0;for(auto z:v){auto e=dec(z,r);if(rank(e)<r)continue;f<<r;for(int a:e)f<<" "<<a;f<<"\n";q++;}cerr<<"fullrank "<<r<<" "<<q<<"\n";tot+=q;}cerr<<"total "<<tot<<"\n";
 }
};
int main(int ac,char**av){if(ac!=5)return 2;int n=stoi(av[1]);E(n).run(stoi(av[2]),stoi(av[3]),av[4]);}
