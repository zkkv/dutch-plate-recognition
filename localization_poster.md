# Localization Process

We initially read all frames and took every 24th one. Then we created a mask using:

- Color information in HSI color space to find yellow(-ish) spots of the image
- Because the mask contained small patches which were definitely not part of the actual plate, we used morphology techniques such as opening with a 3x3 kernel to remove these small patches.
- The next step was to apply median filtering on the mask. This reduced the amount of noise even further.

At this point we have a relatively good mask which occasionally contains other objects such as amber blinkers and/or red tail lights. Sometimes we get the ground selected if it is similar in its tone to yellow number plates.

But nevertheless, this simple algorithm gives surprisingly good results for many images.

In the cropping step, we look at masks' mean values and consider pixels in C * std range, where C is a constant we found by trial-and-error and std is standard deviation. The goal of this is to reduce the influence of pixels that are still present in the image after all the above steps. This gives us min and max values for both axes. Finally, we can use these values to crop the original image and leaves only the number plate.

# Considerations

Our current algorithm does not account for images where two number plates are present at the same time. This is something we should still work on.

The cropping step can definitely get better if all the preceding steps can find the plate more reliably and remove the rest. We are considering using the ratio of the number plates, which for Dutch ones is 51:10, as a way to filter out the shapes which do not fit these proportions. Another thing we are planning to do it set a minimum size threshold (we know that the plate is 100 pixels in width, although to be safe we should probably use a value of ~80 pixels). 

We will experiment more with localization techniques as it is the most crucial and tedious step in the plate recognition procedure.

# Assessment

For each frame we considered intersection-over-union metric on bounding boxes produced by our algorithm and boxes manually annotated by us. Then we averaged this number to get an indication of accuracy for the entire.

[add actual values]
