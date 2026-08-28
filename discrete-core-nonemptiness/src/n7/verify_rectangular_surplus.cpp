
#include <bits/stdc++.h>
#include <boost/multiprecision/cpp_int.hpp>
#include <boost/rational.hpp>
using namespace std;using I=boost::multiprecision::cpp_int;using R=boost::rational<I>;
string rs(R a){return a.numerator().convert_to<string>()+(a.denominator()==1?"":"/"+a.denominator().convert_to<string>());}
struct LP{int m,n;vector<int>B,N;vector<vector<R>>D;LP(vector<vector<R>>const&A,vector<R>const&b,vector<R>const&c):m(b.size()),n(c.size()),B(m),N(n+1),D(m+2,vector<R>(n+2)){for(int i=0;i<m;i++)for(int j=0;j<n;j++)D[i][j]=A[i][j];for(int i=0;i<m;i++)B[i]=n+i,D[i][n]=-1,D[i][n+1]=b[i];for(int j=0;j<n;j++)N[j]=j,D[m][j]=-c[j];N[n]=-1;D[m+1][n]=1;}void piv(int r,int s){R inv=R(1)/D[r][s];for(int i=0;i<m+2;i++)if(i!=r)for(int j=0;j<n+2;j++)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;for(int j=0;j<n+2;j++)if(j!=s)D[r][j]*=inv;for(int i=0;i<m+2;i++)if(i!=r)D[i][s]*=-inv;D[r][s]=inv;swap(B[r],N[s]);}bool simp(int ph){int x=ph==1?m+1:m;while(1){int s=-1;for(int j=0;j<=n;j++){if(ph==2&&N[j]==-1)continue;if(s<0||D[x][j]<D[x][s]||(D[x][j]==D[x][s]&&N[j]<N[s]))s=j;}if(D[x][s]>=0)return 1;int r=-1;for(int i=0;i<m;i++)if(D[i][s]>0){if(r<0)r=i;else{R a=D[i][n+1]/D[i][s],b=D[r][n+1]/D[r][s];if(a<b||(a==b&&B[i]<B[r]))r=i;}}if(r<0)return 0;piv(r,s);}}pair<int,R>solve(){int r=0;for(int i=1;i<m;i++)if(D[i][n+1]<D[r][n+1])r=i;if(D[r][n+1]<0){piv(r,n);if(!simp(1)||D[m+1][n+1]<0)return{-1,0};if(D[m+1][n+1]!=0)return{-1,0};auto it=find(B.begin(),B.end(),-1);if(it!=B.end()){r=it-B.begin();int s=0;for(int j=1;j<=n;j++)if(D[r][j]<D[r][s]||(D[r][j]==D[r][s]&&N[j]<N[s]))s=j;piv(r,s);}}if(!simp(2))return{1,0};return{0,D[m][n+1]};}};
vector<R> solve_linear(vector<vector<R>>M){int n=M.size();for(int c=0;c<n;c++){int p=c;while(p<n&&M[p][c]==0)p++;if(p==n)throw runtime_error("singular");swap(M[p],M[c]);R v=R(1)/M[c][c];for(int j=c;j<=n;j++)M[c][j]*=v;for(int i=0;i<n;i++)if(i!=c&&M[i][c]!=0){R f=M[i][c];for(int j=c;j<=n;j++)M[i][j]-=f*M[c][j];}}vector<R>x(n);for(int i=0;i<n;i++)x[i]=M[i][n];return x;}
struct Cell{int k;vector<int>M;array<int,7>h;vector<vector<int>>A;
 R maxlin(vector<R>obj){int m=6;vector<vector<R>>Q;vector<R>b;auto add=[&](vector<R>r,R z){Q.push_back(move(r));b.push_back(z);};for(int c=0;c<m;c++){vector<R>r(m);r[c]=1;add(r,1);}for(int i=0;i<7;i++){vector<R>r(m);for(int c=0;c<m;c++)r[c]=-A[i][c];add(r,-h[i]);r.assign(m,0);for(int c=0;c<m;c++)r[c]=A[i][c];add(r,h[i]+1);}vector<R>r(m);for(int c=0;c<m;c++)r[c]=1;add(r,k);for(int c=0;c<m;c++)r[c]=-1;add(r,-k);auto z=LP(Q,b,obj).solve();if(z.first!=0)throw runtime_error("closure infeasible");return z.second;}
 R interior_with_sum_ge(int i,int j){int m=6,nv=7,E=6;vector<vector<R>>Q;vector<R>b;auto add=[&](vector<R>r,R z){Q.push_back(move(r));b.push_back(z);};for(int c=0;c<m;c++){vector<R>r(nv);r[c]=-1;r[E]=1;add(r,0);r.assign(nv,0);r[c]=1;r[E]=1;add(r,1);}for(int v=0;v<7;v++){vector<R>r(nv);for(int c=0;c<m;c++)r[c]=-A[v][c];r[E]=1;add(r,-h[v]);r.assign(nv,0);for(int c=0;c<m;c++)r[c]=A[v][c];r[E]=1;add(r,h[v]+1);}vector<R>r(nv);for(int c=0;c<m;c++)r[c]=-(A[i][c]+A[j][c]);add(r,-h[i]-h[j]-1);r.assign(nv,0);for(int c=0;c<m;c++)r[c]=1;add(r,k);for(int c=0;c<m;c++)r[c]=-1;add(r,-k);vector<R>o(nv);o[E]=1;auto z=LP(Q,b,o).solve();return z.first==0?z.second:R(-1);}
};
int main(int ac,char**av){if(ac!=2)return 2;ifstream f(av[1]);string l;int cells=0;while(getline(f,l)){stringstream z(l);char P;int m,k;z>>P>>m>>k;if(m!=6)continue;Cell C;C.k=k;C.M.resize(6);for(int&i:C.M)z>>i;string hs;z>>hs;if(hs[0]=='|')hs=hs.substr(1);for(int i=0;i<7;i++)C.h[i]=hs[i]-'0';C.A.assign(7,vector<int>(6));for(int i=0;i<7;i++)for(int c=0;c<6;c++)C.A[i][c]=(C.M[c]>>i)&1;
 // Find a free alpha coordinate f such that the remaining 6x6 system is nonsingular.
 int free=-1;vector<R>p,q;
 for(int fvar=0;fvar<7&&free<0;fvar++){try{vector<vector<R>>B(6,vector<R>(7));vector<int>vars;for(int i=0;i<7;i++)if(i!=fvar)vars.push_back(i);for(int r=0;r<6;r++){for(int c=0;c<6;c++)B[r][c]=C.A[vars[c]][r];B[r][6]=1;}auto v0=solve_linear(B);for(int r=0;r<6;r++)B[r][6]=1-C.A[fvar][r];auto v1=solve_linear(B);p.assign(7,0);q.assign(7,0);p[fvar]=0;q[fvar]=1;for(int c=0;c<6;c++){p[vars[c]]=v0[c];q[vars[c]]=v1[c]-v0[c];}free=fvar;}catch(...){}}
 if(free<0)throw runtime_error("no free parameter");
 // Positive interval closure [L,U].
 bool hasL=false,hasU=false;R L,U;
 for(int i=0;i<7;i++){if(q[i]>0){R v=-p[i]/q[i];if(!hasL||v>L)L=v,hasL=true;}else if(q[i]<0){R v=-p[i]/q[i];if(!hasU||v<U)U=v,hasU=true;}else if(p[i]<=0)throw runtime_error("empty positive dual");}
 if(!hasL||!hasU||!(L<U))throw runtime_error("bad positive interval");
 int regions=0;
 for(int i=0;i<7;i++){
  R li=L,ui=U;bool ok=1;
  for(int j=0;j<7;j++){R a=q[i]-q[j],b=p[j]-p[i];if(a>0)ui=min(ui,b/a);else if(a<0)li=max(li,b/a);else if(p[i]>p[j])ok=0;}
  if(!ok||li>ui)continue;regions++;
  R amax=max(p[i]+q[i]*li,p[i]+q[i]*ui);
  vector<R>obj(6);for(int c=0;c<6;c++)obj[c]=C.A[i][c];R dmax=C.maxlin(obj)-C.h[i];
  if(!(amax*(dmax+R(k,7))<R(k,7))){cerr<<"single bound failed "<<l<<" i "<<i<<" amax "<<rs(amax)<<" dmax "<<rs(dmax)<<"\n";return 1;}
  for(int j=0;j<7;j++)if(j!=i){for(int c=0;c<6;c++)obj[c]=C.A[i][c]+C.A[j][c];R smax=C.maxlin(obj)-C.h[i]-C.h[j];if(smax<1)continue;if(smax>1){cerr<<"pair sum >1 "<<l<<" i "<<i<<" j "<<j<<" "<<rs(smax)<<"\n";return 1;}R eps=C.interior_with_sum_ge(i,j);if(eps>0){cerr<<"interior pair equality "<<l<<" i "<<i<<" j "<<j<<" eps "<<rs(eps)<<"\n";return 1;}}
 }
 if(!regions)throw runtime_error("no minimizer regions");cells++;}
 cerr<<"rectangular_cells "<<cells<<" verified_for_all_positive_duals_and_interior_points\n";}
