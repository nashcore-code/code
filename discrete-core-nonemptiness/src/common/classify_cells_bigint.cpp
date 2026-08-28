
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
using namespace std;
using BI=boost::multiprecision::cpp_int;
using Q=boost::rational<BI>;
string qs(Q const&a){return a.numerator().convert_to<string>()+(a.denominator()==1?"":"/"+a.denominator().convert_to<string>());}
struct LP{
 int m,n;vector<int>B,N;vector<vector<Q>>D;
 LP(vector<vector<Q>>const&A,vector<Q>const&b,vector<Q>const&c):m(b.size()),n(c.size()),B(m),N(n+1),D(m+2,vector<Q>(n+2)){
  for(int i=0;i<m;i++)for(int j=0;j<n;j++)D[i][j]=A[i][j];
  for(int i=0;i<m;i++)B[i]=n+i,D[i][n]=Q(-1),D[i][n+1]=b[i];
  for(int j=0;j<n;j++)N[j]=j,D[m][j]=-c[j];N[n]=-1;D[m+1][n]=Q(1);
 }
 void piv(int r,int s){Q inv=Q(1)/D[r][s];for(int i=0;i<m+2;i++)if(i!=r)for(int j=0;j<n+2;j++)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;for(int j=0;j<n+2;j++)if(j!=s)D[r][j]*=inv;for(int i=0;i<m+2;i++)if(i!=r)D[i][s]*=-inv;D[r][s]=inv;swap(B[r],N[s]);}
 bool simp(int ph){int x=ph==1?m+1:m;while(1){int s=-1;for(int j=0;j<=n;j++){if(ph==2&&N[j]==-1)continue;if(s<0||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&N[j]<N[s]))s=j;}if(D[x][s]>=Q(0))return true;int r=-1;for(int i=0;i<m;i++)if(D[i][s]>Q(0)){if(r<0)r=i;else{Q a=D[i][n+1]/D[i][s],b=D[r][n+1]/D[r][s];if(a<b||(a==b&&B[i]<B[r]))r=i;}}if(r<0)return false;piv(r,s);}}
 // status -1 infeasible, 0 optimal, 1 unbounded
 pair<int,Q> solve(vector<Q>&x){int r=0;for(int i=1;i<m;i++)if(D[i][n+1]<D[r][n+1])r=i;if(D[r][n+1]<Q(0)){piv(r,n);if(!simp(1)||D[m+1][n+1]<Q(0))return{-1,Q()};if(D[m+1][n+1]!=Q(0))return{-1,Q()};auto it=find(B.begin(),B.end(),-1);if(it!=B.end()){r=it-B.begin();int s=0;for(int j=1;j<=n;j++)if(D[r][j]<D[r][s]||(D[r][j]==D[r][s]&&N[j]<N[s]))s=j;piv(r,s);}}if(!simp(2))return{1,Q()};x.assign(n,Q());for(int i=0;i<m;i++)if(B[i]<n)x[B[i]]=D[i][n+1];return{0,D[m][n+1]};}
};
struct CellLP{
 int n,m,k;vector<int>M;vector<array<int,7>>col;
 CellLP(int nn,int kk,vector<int>mm):n(nn),m(mm.size()),k(kk),M(move(mm)),col(m){for(int c=0;c<m;c++)for(int i=0;i<n;i++)col[c][i]=(M[c]>>i)&1;}
 Q maxeps(array<int,7>const&h,vector<int>const&strict,int equalrow=-1){
  int nv=m+1,E=m;vector<vector<Q>>A;vector<Q>b;
  auto add=[&](vector<Q>r,Q z){A.push_back(move(r));b.push_back(z);};
  for(int c=0;c<m;c++){vector<Q>r(nv);r[c]=Q(-1);r[E]=Q(1);add(r,Q(0));r.assign(nv,Q());r[c]=Q(1);r[E]=Q(1);add(r,Q(1));}
  vector<int>isstr(n);for(int i:strict)isstr[i]=1;
  for(int i=0;i<n;i++){vector<Q>r(nv);for(int c=0;c<m;c++)r[c]=Q(-col[c][i]);add(r,Q(-h[i]));r.assign(nv,Q());for(int c=0;c<m;c++)r[c]=Q(col[c][i]);r[E]=Q(1);add(r,Q(h[i]+1));if(isstr[i]){r.assign(nv,Q());for(int c=0;c<m;c++)r[c]=Q(-col[c][i]);r[E]=Q(1);add(r,Q(-h[i]));}}
  if(equalrow>=0){vector<Q>r(nv);for(int c=0;c<m;c++)r[c]=Q(col[c][equalrow]);add(r,Q(h[equalrow]));}
  vector<Q>r(nv);for(int c=0;c<m;c++)r[c]=Q(1);add(r,Q(k));for(int c=0;c<m;c++)r[c]=Q(-1);add(r,Q(-k));
  vector<Q>obj(nv); obj[E]=Q(1); vector<Q>x; LP lp(A,b,obj); auto z=lp.solve(x);if(z.first!=0)return Q(-1);return z.second;
 }
 vector<array<int,7>> utilities()const{vector<array<int,7>>U;vector<int>ch;function<void(int,int,array<int,7>)>rec=[&](int st,int q,array<int,7>u){if(!q){U.push_back(u);return;}for(int c=st;c<=m-q;c++){auto v=u;for(int i=0;i<n;i++)v[i]+=col[c][i];rec(c+1,q-1,v);}};array<int,7>z{};rec(0,k,z);return U;}
};
int main(int ac,char**av){if(ac!=5)return 2;int n=stoi(av[1]);ifstream f(av[2]);ofstream bad(av[3]),sur(av[4]);string l,last;unique_ptr<CellLP>C;vector<array<int,7>>U;long long lines=0,prim=0,nonprim=0,surplus=0,badn=0,lps=0;while(getline(f,l)){lines++;stringstream aa(l);char flag;aa>>flag;string rest;getline(aa,rest);auto bar=rest.find('|');string L=rest.substr(0,bar),hs=rest.substr(bar+1);stringstream ss(L);int m,k;ss>>m>>k;vector<int>M(m);for(int&i:M)ss>>i;string key=to_string(m)+","+to_string(k);for(int z:M)key+=","+to_string(z);if(key!=last){C=make_unique<CellLP>(n,k,M);U=C->utilities();last=key;}array<int,7>h{};for(int i=0;i<n;i++)h[i]=hs[i]-'0';vector<int>B;for(int i=0;i<n;i++)if(h[i]>0){auto t=h;t[i]--;bool ok=0;for(auto u:U){bool yes=1;for(int j=0;j<n;j++)if(u[j]<t[j]){yes=0;break;}if(yes){ok=1;break;}}if(ok)B.push_back(i);}
 bool fail=0;
 if(flag=='N'){nonprim++;Q e=C->maxeps(h,B);lps++;if(e>Q(0)){bad<<l<<" eps="<<qs(e)<<"\n";fail=1;}}
 else{prim++;vector<int>all(n);iota(all.begin(),all.end(),0);Q e=C->maxeps(h,all);lps++;if(e>Q(0)){sur<<l<<" eps="<<qs(e)<<" B=";for(int i:B)sur<<i;sur<<"\n";surplus++;}
  vector<int>inB(n);for(int i:B)inB[i]=1;for(int q=0;q<n&&!fail;q++)if(!inB[q]){Q z=C->maxeps(h,B,q);lps++;if(z>Q(0)){bad<<l<<" boundary="<<q<<" eps="<<qs(z)<<"\n";fail=1;}}
 }
 if(fail)badn++;
 if(lines%1000==0)cerr<<"lines "<<lines<<" lps "<<lps<<" surplus "<<surplus<<" bad "<<badn<<"\n";
 }cerr<<"lines "<<lines<<" primitive "<<prim<<" nonprimitive "<<nonprim<<" lps "<<lps<<" surplus "<<surplus<<" bad "<<badn<<"\n";}
