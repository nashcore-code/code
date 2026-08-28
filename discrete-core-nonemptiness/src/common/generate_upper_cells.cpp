
#include <bits/stdc++.h>
using namespace std;using V=array<unsigned char,7>;
uint32_t enc(V const&h,int n){uint32_t z=0;for(int i=0;i<n;i++)z|=uint32_t(h[i])<<(3*i);return z;}V dec(uint32_t z,int n){V h{};for(int i=0;i<n;i++)h[i]=(z>>(3*i))&7;return h;}
int main(int ac,char**av){if(ac!=4)return 2;int n=stoi(av[1]);ifstream f(av[2]);ofstream o(av[3]);string l,last;vector<V>mins;vector<int>M;int m=0,k=0;long long G=0,C=0;
 auto flush=[&](){if(mins.empty())return;G++;V up{};for(int i=0;i<n;i++){int d=0;for(int a:M)d+=(a>>i)&1;up[i]=d?d-1:0;}unordered_set<uint32_t>S,P;for(auto g:mins){P.insert(enc(g,n));V h=g;function<void(int)>rec=[&](int i){if(i==n){S.insert(enc(h,n));return;}for(int z=g[i];z<=up[i];z++){h[i]=z;rec(i+1);}};rec(0);}vector<uint32_t>v(S.begin(),S.end());sort(v.begin(),v.end());for(auto z:v){auto h=dec(z,n);o<<(P.count(z)?"P ":"N ")<<m<<" "<<k;for(int a:M)o<<" "<<a;o<<" |";for(int i=0;i<n;i++)o<<int(h[i]);o<<"\n";}C+=S.size();mins.clear();};
 while(getline(f,l)){auto b=l.find('|');string L=l.substr(0,b),hs=l.substr(b+1);stringstream ss(L);int mm,kk;ss>>mm>>kk;vector<int>X(mm);for(int&i:X)ss>>i;string key=to_string(mm)+","+to_string(kk);for(int a:X)key+=","+to_string(a);if(!last.empty()&&key!=last)flush();if(key!=last){m=mm;k=kk;M=X;last=key;}V h{};for(int i=0;i<n;i++)h[i]=hs[i]-'0';mins.push_back(h);}flush();cerr<<"groups "<<G<<" upper_cells "<<C<<"\n";}
