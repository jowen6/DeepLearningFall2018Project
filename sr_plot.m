clc
clear

filename = './results/success_rate.csv';
vec = importdata(filename);

filename = './results/success_rate_2.csv';
vec_2 = importdata(filename);

title('Learning rate \lambda = 0.001')
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

legend('Original test set','Enriched test set','Location','northwest')


filename = './results/confidence_rate.csv';
vec = importdata(filename);

filename = './results/confidence_rate_2.csv';
vec_2 = importdata(filename);

figure
grid on
grid minor
xticks(0:40:400)
axis([0,400,50,100])
hold on

title('Learning rate \lambda = 0.001')
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
    'Location','northwest')

