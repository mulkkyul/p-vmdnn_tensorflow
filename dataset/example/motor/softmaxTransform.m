clear all;
clc; clf;

%% Specify the idx of data
targetList = [
1000
1055
    ];


numberOfTrainingPattern = size(targetList,1);
numberOfJoints = 2;

%% Softmax Parameters
softmaxUnit = 10;
minVal = 15-15; maxVal = 105+15; %% Add a bit of maring
SIGMA = 150;

mkdir('./softmax');
for idxTraj = 1:numberOfTrainingPattern
    idxTarget = targetList(idxTraj,1);
    %% SOFTMAX TRANSFORM
    traj = load(sprintf('./analog/target_JNT3_LR_%04d.txt',idxTarget));
    fid_s = fopen(sprintf('./softmax/target_%d_softmax.txt',idxTarget),'w');
    for idxStep = 1:size(traj,1)
        for idxJnt = 1:size(traj,2)
            references = linspace(minVal,maxVal,softmaxUnit);
            val = zeros(1,softmaxUnit);
            sumVal = 0;
            for idxRef = 1:softmaxUnit
                val(1,idxRef) = power((references(1,idxRef) - traj(idxStep,idxJnt)),2);
                val(1,idxRef) = exp(-val(1,idxRef) / SIGMA);
                sumVal = sumVal + val(1,idxRef);
            end
            for idxRef = 1:softmaxUnit
                val(1,idxRef) = val(1,idxRef) / sumVal;
                fprintf(fid_s,'%.6f\t',val(1,idxRef));
            end
        end
        fprintf(fid_s,'\n');
    end
    fclose(fid_s);
    
    %% INVERSE TRANSFORM
    softmax = load(sprintf('./softmax/target_%d_softmax.txt',idxTarget));
    fid_i = fopen(sprintf('./softmax/target_%d_inverse.txt',idxTarget),'w');
    
    analogJnt = 1;
    analog = zeros(size(softmax,1),size(softmax,2)/softmaxUnit);
    for idxJnt = 1:softmaxUnit:size(softmax,2)
        references = linspace(minVal,maxVal,softmaxUnit);
        analog(:,analogJnt) = softmax(:,idxJnt:idxJnt+softmaxUnit-1) * transpose(references);
        analogJnt = analogJnt + 1;
    end
    
    for idxStep = 1:size(softmax,1)
        for idxJnt = 1:size(analog,2)
            fprintf(fid_i,'%.6f\t',analog(idxStep,idxJnt));
        end
        fprintf(fid_i,'\n');
    end
    fclose(fid_i);
    
    
    %% COMP
    traj = load(sprintf('./analog/target_JNT3_LR_%04d.txt',idxTarget));
    inverse = load(sprintf('./softmax/target_%d_inverse.txt',idxTarget));
    
    cmap = hsv(size(traj,2));
    t = linspace(1,size(traj,1),size(traj,1));
    figure(1)
    clf;
    for idxJnt = 1:size(traj,2)
        plot(t,traj(:,idxJnt),'color',cmap(idxJnt,:));
        hold on;
        plot(t,inverse(:,idxJnt),'--','color',cmap(idxJnt,:));
        hold on;
        ylim([-minVal maxVal])
    end
    MSEs = mean((traj - inverse).^2);
    figTitle = sprintf('IDX: %d \t MSE: %.5f',idxTarget, sum(MSEs));
    suptitle(figTitle);
    waitforbuttonpress
end

%% Just to check the softmax output
if(1)
    clf;
    for idxTraj = 0:numberOfTrainingPattern-1
        softmax = load(sprintf('./softmax/target_%d_softmax.txt',idxTarget));
        idxAnalog = 1;
        for idxJnt = 1:softmaxUnit:size(softmax,2)
            vals = softmax(:,idxJnt:idxJnt+softmaxUnit-1);
            for idxStep = 1:size(vals,1)
                bar(vals(idxStep,:));
                suptitle(sprintf('Dim: %d \t Step: %04d',idxAnalog, idxStep));
                ylim([0 1]);
                pause(0.01);
            end
            idxAnalog = idxAnalog + 1;
        end
    end
end
