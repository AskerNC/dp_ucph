clear
mp.nC=4;
mp=lf.setup(mp)

%% ------------- Initialize data containers ----------------
ss = lf.cSS(mp.nC);
ESS = lf.cESS(mp.nC);

%% test
[ ss , ESS ] = lf.solve_last_corner( ss , ESS ,mp)
[ ss , ESS ] = lf.solve_last_edge( ss , ESS , mp)
[ ss , ESS ] = lf.solve_last_interior( ss , ESS , mp)

ss(4).nEQ

ss(4).EQs(4,4,1).eq

ss(4).EQs(4,4,2).eq

ss(4).EQs(1,2,2).eq

ESS.esr
ESS.bases

[ ss , ESS ] = lf.solve_corner(ss,3,ESS,mp)


%%

Gtau= @(ss, ESS, tau) lf.state_recursion(ss,ESS, tau, mp);    
[ESS, TAU, out]=rls.solve(Gtau,ss,ESS,mp.stage_index);

number_of_equilibria=size(TAU, 1);
T=numel(mp.stage_index);

y = zeros(T,1);
for i = 1:T
    y(i) = sum(TAU==i);
end

for iEQ=1:number_of_equilibria
    V(iEQ,1)=out(iEQ).V1;
    V(iEQ,2)=out(iEQ).V2;
    MPEesr(iEQ,:)=out(iEQ).MPEesr; 
end

bar(y);
array2table([ (1:T)' , y],'VariableNames',{'Stage','Recursion_started_in_stage'})
number_of_equilibria
lf.vscatter(V,1,0.05,1);



