
#include <bits/stdc++.h>
using namespace std;
struct LP{
 int m,n;vector<int>B,N;vector<vector<double>>D;static constexpr double EPS=1e-10,INF=1e100;
 LP(vector<vector<double>>const&A,vector<double>const&b,vector<double>const&c):m(b.size()),n(c.size()),B(m),N(n+1),D(m+2,vector<double>(n+2)){
  for(int i=0;i<m;i++)for(int j=0;j<n;j++)D[i][j]=A[i][j];
  for(int i=0;i<m;i++)B[i]=n+i,D[i][n]=-1,D[i][n+1]=b[i];
  for(int j=0;j<n;j++)N[j]=j,D[m][j]=-c[j];N[n]=-1;D[m+1][n]=1;
 }
 void piv(int r,int s){double inv=1.0/D[r][s];for(int i=0;i<m+2;i++)if(i!=r)for(int j=0;j<n+2;j++)if(j!=s)D[i][j]-=D[r][j]*D[i][s]*inv;for(int j=0;j<n+2;j++)if(j!=s)D[r][j]*=inv;for(int i=0;i<m+2;i++)if(i!=r)D[i][s]*=-inv;D[r][s]=inv;swap(B[r],N[s]);}
 bool simp(int ph){int x=ph==1?m+1:m;while(1){int s=-1;for(int j=0;j<=n;j++){if(ph==2&&N[j]==-1)continue;if(s==-1||D[x][j]<D[x][s]-EPS||(abs(D[x][j]-D[x][s])<=EPS&&N[j]<N[s]))s=j;}if(D[x][s]>=-EPS)return true;int r=-1;for(int i=0;i<m;i++)if(D[i][s]>EPS){if(r==-1)r=i;else{double a=D[i][n+1]/D[i][s],b=D[r][n+1]/D[r][s];if(a<b-EPS||(abs(a-b)<=EPS&&B[i]<B[r]))r=i;}}if(r==-1)return false;piv(r,s);}}
 double solve(vector<double>&x){int r=0;for(int i=1;i<m;i++)if(D[i][n+1]<D[r][n+1])r=i;if(D[r][n+1]<-EPS){piv(r,n);if(!simp(1)||D[m+1][n+1]<-EPS)return -INF;if(abs(D[m+1][n+1])>EPS)return -INF;if(find(B.begin(),B.end(),-1)!=B.end()){r=find(B.begin(),B.end(),-1)-B.begin();int s=0;for(int j=1;j<=n;j++)if(D[r][j]<D[r][s]-EPS||(abs(D[r][j]-D[r][s])<=EPS&&N[j]<N[s]))s=j;piv(r,s);}}if(!simp(2))return INF;x.assign(n,0);for(int i=0;i<m;i++)if(B[i]<n)x[B[i]]=D[i][n+1];return D[m][n+1];}
};
int main(int ac,char**av){if(ac!=4)return 2;int nv=stoi(av[1]);ifstream f(av[2]);ofstream o(av[3]);string l,key,last;vector<vector<double>>AA;vector<double>bb;int m0=0;long long in=0,out=0,cert=0;while(getline(f,l)){in++;auto bar=l.find('|');string L=l.substr(0,bar),hs=l.substr(bar+1);stringstream ss(L);int m,k;ss>>m>>k;vector<int>M(m);for(int&i:M)ss>>i;key=to_string(m);for(int z:M)key+=","+to_string(z);if(key!=last){m0=m;AA.assign(m,vector<double>(nv+m));bb.assign(m,1);for(int c=0;c<m;c++){for(int i=0;i<nv;i++)AA[c][i]=(M[c]>>i)&1;AA[c][nv+c]=-1;}last=key;}vector<double>c(nv+m);for(int i=0;i<nv;i++)c[i]=hs[i]-'0';for(int j=0;j<m;j++)c[nv+j]=-1;vector<double>x;LP lp(AA,bb,c);double val=lp.solve(x);bool rejected=false;if(val>k+1e-8&&val<LP::INF/2){for(long long DEN: {1000000LL,1000000000LL}){vector<long long>lam(nv);bool safe=true;for(int i=0;i<nv;i++){double z=max(0.0,x[i]);if(!isfinite(z)||z*DEN>9e18){safe=false;break;}lam[i]=llround(z*DEN);if(lam[i]<0)lam[i]=0;}if(!safe)continue;__int128 num=0;for(int i=0;i<nv;i++)num+=(__int128)(hs[i]-'0')*lam[i];for(int j=0;j<m;j++){long long load=0;for(int i=0;i<nv;i++)if((M[j]>>i)&1)load+=lam[i];if(load>DEN)num-=load-DEN;}if(num>(__int128)k*DEN){rejected=true;cert++;break;}}}if(!rejected){o<<l<<"\n";out++;}}cerr<<"input "<<in<<" retained "<<out<<" exact_dual_rejections "<<cert<<"\n";}
