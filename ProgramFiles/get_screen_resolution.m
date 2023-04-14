function [width, height] = get_screen_resolution()
    set(0,'units','pixels')
    res = get(0,'ScreenSize');
    res = res(3:4);
    width = res(1);
    height = res(2);
end

