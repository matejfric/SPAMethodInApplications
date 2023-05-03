function [] = save_animation_gif(frames, filename, fps)
%SAVE_ANIMATION_GIF
arguments
    frames
    filename="my_animation.gif"
    fps = 50
end

%{
   The highest fps possible representable
   in the GIF file format specification is 100 fps. 
   However most gif viewers support only 60 fps 
   (including the default one on Windows).
   Browsers in 2023 support max 50 fps.
   Source: https://wunkolo.github.io/post/2020/02/buttery-smooth-10fps/
%}

f = frames(1);
imwrite(f.cdata,f.colormap,filename,'gif', 'Loopcount',inf,'DelayTime',1/fps);

for k = progress(2:length(frames))
    f = frames(k);
    imwrite(f.cdata,f.colormap,filename,'gif','WriteMode','append',...
    'DelayTime',1/fps); 
end

end

