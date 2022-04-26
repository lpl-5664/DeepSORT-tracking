videoReader = VideoReader("C:/PhiLongLai/University/4th_year/Winter_2022/CMPUT428/Final_project/DeepSORT-tracking/test/people_walking.mp4");
videoPlayer = vision.VideoPlayer('Position',[100,100,1200,700]);
initFrame = readFrame(videoReader);

figure;
hold on;
imshow(initFrame);
rect = getrect;

x = int16(rect(1));
y = int16(rect(2));
width = int16(rect(3));
height = int16(rect(4));
hold off;
close;

origin = initFrame;
originPyramid = cell(4,1);
originPyramid{1} = origin;
for index=2:4
    origin = impyramid(origin, 'reduce');
    originPyramid{index} = origin;
end
p = zeros(2,1);

video = VideoWriter('highway_LK_tracking.avi','Uncompressed AVI');
open(video);

while hasFrame(videoReader)
    frame = readFrame(videoReader);
    image = frame;
    imagePyramid = cell(4, 1);
    imagePyramid{1} = image;
    for index=2:4
        image = impyramid(image, 'reduce');
        imagePyramid{index} = image;
    end
    p_pyramid = zeros(2,1);

    for level=3:-1:0
        templateX = int16(x/2^level);
        templateY = int16(y/2^level);
        newWidth = int16(width/2^level);
        newHeight = int16(height/2^level);
        template = originPyramid{level+1}(templateY:templateY+newWidth, ...
            templateX:templateX+newHeight);
        [dU, dV] = gradient(double(template));
        dUv = dU(:);
        dVv = dV(:);
        M = [dUv dVv];

        p_LK = zeros(2,1);
        deltaP = ones(2,1);
        prev_norm = norm(deltaP);
        iter = 0;
        while (norm(deltaP) >= 0.1*prev_norm) && (iter < 15)
            prev_norm = norm(deltaP);
            trackX = int16(templateX+p(1)/(2^level)+p_pyramid(1)+p_LK(1));
            trackY = int16(templateY+p(2)/(2^level)+p_pyramid(2)+p_LK(2));
            track_box = imagePyramid{level+1}(trackY:trackY+newWidth, ...
                trackX:trackX+newHeight);
        
            dI = double(track_box - template);
            dIv = -dI(:);
            deltaP = M\dIv;
            p_LK = p_LK + deltaP;
            
            iter = iter + 1;
        end
        if level == 0
            p_pyramid = p_pyramid + p_LK;
        else
            p_pyramid = 2*(p_pyramid + p_LK);
        end
    end
    p = p + p_pyramid;
    X = templateX + p(1);
    Y = templateY + p(2);

    out = insertShape(frame,"Rectangle",[X Y newWidth newHeight],"Color","blue");
    writeVideo(video, out);
    videoPlayer(out);
end
close(video);
release(videoPlayer);