% change this path if you install the VOC code elsewhere
addpath([cd '/VOCcode']);

% initialize VOC options
VOCinit;

% ------- set the image mean file here ---------- %
data_mean_file = 'ilsvrc_2012_mean.mat';
assert(exist(data_mean_file, 'file') ~= 0);
ld = load(data_mean_file);
image_mean = ld.mean_data; clear ld;

imdir = 'VOC2012/JPEGImages/%s';
for i=2:VOCopts.nactions % skip "other"
    cls=VOCopts.actions{i};    
    [imgids,objids,gt]=textread(sprintf(VOCopts.action.clsimgsetpath,cls,'train'),'%s %d %d');
    [imgids2,objids2,gt2]=textread(sprintf(VOCopts.action.clsimgsetpath,cls,'val'),'%s %d %d');
    imgids = cat(1,imgids,imgids2);
    objids = cat(1,objids, objids2);
    gt = cat(1,gt,gt2);
    rp = randperm(length(gt));
    imgids = imgids(rp);
    objids = objids(rp);
    gt = gt(rp);
    
    tic;
    train_num = floor(length(imgids)/3);
    val_num = train_num;
    test_num = length(imgids) - train_num * 2;
    % split for training    
    infoname = ['random_action/VOC2012_action_train_' cls '_info.dat'];
    imname = ['random_action/VOC2012_action_train_' cls '_ims.dat'];
    finfo = fopen(infoname,'w');
    fim = fopen(imname,'wb');    
    for k=1:train_num    
        % display progress
        if toc>1
            fprintf('%s: train: %d/%d\n',cls,k,length(imgids));
            drawnow;
            tic;
        end

        rec=PASreadrecord(sprintf(VOCopts.annopath,imgids{k}));
        obj=rec.objects(objids(k));
        fprintf(finfo,'%s %d %d\n', rec.filename(1:end-4), objids(k), gt(k));
        x1 = obj.bndbox.xmin;
        y1 = obj.bndbox.ymin;
        x2 = obj.bndbox.xmax;
        y2 = obj.bndbox.ymax;
        bbox = [x1 y1 x2 y2];       
        impath = sprintf(imdir,rec.filename);
        im = imread(impath);
        im = single(im(:,:,[3 2 1]));
        crop = rcnn_im_crop(im, bbox, 'wrap', 227, ...
            16, image_mean); 
        crop = permute(crop, [2 1 3]);
        % could be negative due to mean subtraction
        fwrite(fim, crop, 'int8');
    end
    fclose(finfo);
    fclose(fim);
    
    % split for val    
    infoname = ['random_action/VOC2012_action_val_' cls '_info.dat'];
    imname = ['random_action/VOC2012_action_val_' cls '_ims.dat'];
    finfo = fopen(infoname,'w');
    fim = fopen(imname,'wb');    
    for k=(train_num+1):(train_num+val_num)
        % display progress
        if toc>1
            fprintf('%s: val: %d/%d\n',cls,k,length(imgids));
            drawnow;
            tic;
        end

        rec=PASreadrecord(sprintf(VOCopts.annopath,imgids{k}));
        obj=rec.objects(objids(k));
        fprintf(finfo,'%s %d %d\n', rec.filename(1:end-4), objids(k), gt(k));
        x1 = obj.bndbox.xmin;
        y1 = obj.bndbox.ymin;
        x2 = obj.bndbox.xmax;
        y2 = obj.bndbox.ymax;
        bbox = [x1 y1 x2 y2];       
        impath = sprintf(imdir,rec.filename);
        im = imread(impath);
        im = single(im(:,:,[3 2 1]));
        crop = rcnn_im_crop(im, bbox, 'wrap', 227, ...
            16, image_mean); 
        crop = permute(crop, [2 1 3]);
        % could be negative due to mean subtraction
        fwrite(fim, crop, 'int8');
    end
    fclose(finfo);
    fclose(fim);

    % test        
    infoname = ['random_action/VOC2012_action_test_' cls '_info.dat'];
    imname = ['random_action/VOC2012_action_test_' cls '_ims.dat'];
    finfo = fopen(infoname,'w');
    fim = fopen(imname,'wb');    
    for k = (train_num + val_num +1) : length(imgids)
        % display progress
        if toc>1
            fprintf('%s: test: %d/%d\n',cls,k,length(imgids));
            drawnow;
            tic;
        end
        rec=PASreadrecord(sprintf(VOCopts.annopath,imgids{k}));
        obj=rec.objects(objids(k));
        fprintf(finfo,'%s %d %d\n', rec.filename(1:end-4), objids(k), gt(k));
        x1 = obj.bndbox.xmin;
        y1 = obj.bndbox.ymin;
        x2 = obj.bndbox.xmax;
        y2 = obj.bndbox.ymax;
        bbox = [x1 y1 x2 y2];       
        impath = sprintf(imdir,rec.filename);
        im = imread(impath);
        im = single(im(:,:,[3 2 1]));
        crop = rcnn_im_crop(im, bbox, 'wrap', 227, ...
            16, image_mean); 
        crop = permute(crop, [2 1 3]);
        % could be negative due to mean subtraction
        fwrite(fim, crop, 'int8');
    end
    fclose(finfo);
    fclose(fim);    
end
