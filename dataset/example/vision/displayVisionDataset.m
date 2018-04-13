clear all; clc; clf;
%% Display Training Dataset
trial = load('../trialList.txt');
for i = 1:size(trial,1)
    target = trial(i,1);
    vision = load(sprintf('./vision_%04d.txt',target));
    for s = 1:size(vision,1)
        tempFrame = (transpose(reshape(vision(s,:),[64 48]))+1)./2;
        imshow(tempFrame,'InitialMagnification',500)
        title(sprintf('File: %04d \t Step: %04d',i,s))
        drawnow
    end
end

%% Dislay Testing Dataset
vision = load('./data_er_0000.txt');
for s = 1:size(vision,1)
    tempFrame = (transpose(reshape(vision(s,:),[64 48]))+1)./2;
    imshow(tempFrame,'InitialMagnification',500)
    title(sprintf('Testing File \t Step: %04d',s))
    drawnow
end