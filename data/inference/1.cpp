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
	int t,a,b,T;
	T=fr();
	while(T--){
		t=fr();
		a=t%11,b=t/11;
		if(a<=b/10) puts("YES\n");
		else puts("NO\n");
	}
	return 0;
}