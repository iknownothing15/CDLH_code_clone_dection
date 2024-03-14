int n,a,b,T;
inline int fr(){
	int res=0;char tp=getchar();
	while(!isdigit(tp)){
		tp=getchar();
	}
	while(isdigit(tp)){
		res=(res<<1)+(res<<3)+tp-'0';
		tp=getchar();
	}
	return res;
}

int main(){
	
	T=fr();
	
	while(T--){
		n=fr();
		a=n%11,b=n/11;
		if(a<=b/10) puts("YES\n");
		else puts("NO\n");
	}
	
	return 0;
}