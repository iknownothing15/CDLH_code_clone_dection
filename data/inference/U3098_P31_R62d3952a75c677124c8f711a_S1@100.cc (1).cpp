#include<stdio.h>
#include<string.h>
#define MOD 998244353;

long long n;

void k1(int *f,int d[3][3]){
	int t[3];
	memset(t,0,sizeof(t));
	for(int j=0;j<3;j++)
	   for(int k=0;k<3;k++)
	      t[j]=(t[j]+(long long)f[k]*d[k][j])%MOD;
	memcpy(f,t,sizeof(t));
}

void k2(int d[3][3]){
	int t[3][3];
	memset(t,0,sizeof(t));
	for(int i=0;i<3;i++)
	   for(int j=0;j<3;j++)
	      for(int k=0;k<3;k++)
	         t[i][j]=(t[i][j]+(long long)d[i][k]*d[k][j])%MOD;
	memcpy(d,t,sizeof(t));
}
int main(){
	
	scanf("%lld",&n);
	
	int f[3]={1,1,1};
	int a[3][3]={{0,0,1},{1,0,1},{0,1,1}};
    n--;
    
    while(n){
    	if(n&1) k1(f,a);
    	k2(a);
    	n>>=1;
	}
	
	printf("%d",f[0]);
	return 0;
}