function [z,delT]=integrate_fullKZ(N,M,Nskip)


% integrate full Kac-Zwanzig model to generate data

% follow set-up from Stuart and Warren, J. Stat. Phys. 1999

% initial Nskip*10^3 integration steps are discarded as transient
    
% N = number of particles in the heat bath
% M = number of output datapoints
% Nskip = output every Nskip integration steps
% delT = output time step, delT = dt*Nskip = Nskip*0.01/N


g=1; % coupling and mass scaling (gamma in the paper)
b=0.0001;   % inverse temperature (beta in the paper)

g2=g^2;

j2=zeros(N,1);
for j=1:N
    j2(j)=j^2;
end

% initialisation:

% distinguished particle:

q=1;  % position
p=0;  % momentum

% heat bath particles:

ui=randn(N,1);

u=ui/(g*sqrt(b)); % position
v=zeros(N,1);    % momentum


% potential V(q)=1/4 * (q^2-1)^2

% integration with symplectic Euler:

dt=0.01/N;  % integration time step

delT = dt*Nskip;  % output time step

z=zeros(3,M);


for j=1:Nskip*1e3

    v=v-dt*j2.*(u-q);
    p=p-dt*q*(q^2-1)+dt*g2*(sum(u)-N*q);
    u=u+dt*v;
    q=q+dt*p;

end


for i=1:M
    for j=1:Nskip
        
        v=v-dt*j2.*(u-q);
        p=p-dt*q*(q^2-1)+dt*g2*(sum(u)-N*q);
        u=u+dt*v;
        q=q+dt*p;
                
    end
    
    z(1,i)=q;
    z(2,i)=p;
    z(3,i)=sum(u);   % sum(u) =: r
    
end

