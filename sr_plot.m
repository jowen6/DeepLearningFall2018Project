clc
clear

filename = './results/success_rate.csv';
vec = importdata(filename);

filename = './results/success_rate_2.csv';
vec_2 = importdata(filename);

filename = './results/success_rate_lr_0_01.csv';
vec_3 = importdata(filename);

filename = './results/success_rate_lr_0_1.csv';
vec_4 = importdata(filename);

xlabel('Training data set size M')
ylabel('Success rate %')
ax = gca;
ax.FontSize = 12;
ytickformat(ax, 'percentage');

hold on
grid on
grid minor
xticks(0:40:400)
axis([0,400,50,100])

plot(vec(:,1), 100*vec(:,2),'o-')
plot(vec_2(:,1), 100*vec_2(:,2),'d-')
plot(vec_3(:,1), 100*vec_3(:,2),'s-')
plot(vec_4(:,1), 100*vec_4(:,2),'s-')

legend('Original test set \lambda=0.001','Enriched test set \lambda=0.001', ...
'Enriched test set \lambda=0.01','Enriched test set \lambda=0.1',...
'Location','southeast')


%% ===================================================================
filename = './results/confidence_rate.csv';
vec = importdata(filename);

filename = './results/confidence_rate_2.csv';
vec_2 = importdata(filename);

filename = './results/confidence_rate_lr_0_01.csv';
vec_3 = importdata(filename);

figure
grid on
grid minor
xticks(0:40:400)
axis([0,400,50,100])
hold on

xlabel('Training data set size M')
ylabel('Confidence rate %')
ax = gca;
ax.FontSize = 12;
ytickformat(ax, 'percentage');

plot(vec(:,1), 100*(1-vec(:,2)),'o-' )
plot(vec(:,1), 100*(1-vec(:,3)),'d-' )
plot(vec_2(:,1), 100*(1-vec_2(:,2)),'<-' )
plot(vec_2(:,1), 100*(1-vec_2(:,3)),'>-' )

legend('Original test #1','Original test #2',...
    'Enriched test #1','Enriched test #2',...
    'Location','southeast')

