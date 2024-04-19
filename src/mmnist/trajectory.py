import random

import torchvision.transforms.functional as TF


class BaseTrajectory:
    def __init__(
        self, digit_label, affine_params, n, padding, initial_position, **kwargs
    ):
        self.digit_label = digit_label
        self.affine_params = affine_params
        self.n = n
        self.padding = padding
        self.position = initial_position
        self.kwargs = kwargs

        # Set fixed initial values for the transformation
        self.translate = (
            random.randint(*self.affine_params.translate[0]),
            random.randint(*self.affine_params.translate[1]),
        )
        if self.translate[0] == 0:
            self.translate = (self.translate[0] + 1, self.translate[1])
        if self.translate[1] == 0:
            self.translate = (self.translate[0], self.translate[1] + 1)
        self.angle = random.uniform(*self.affine_params.angle)
        self.scale = random.uniform(*self.affine_params.scale)
        self.shear = random.uniform(*self.affine_params.shear)

    def transform(self, img):
        raise NotImplementedError("This method should be implemented by subclasses.")

    def __call__(self, img):
        sequence = [img]
        transformations = []
        for _ in range(self.n):
            img, caption = self.transform(sequence[-1])
            sequence.append(img)
            transformations.append(caption)
        return sequence, transformations

    def describe_movement(self, translate, scale):
        horizontal_direction = "right" if translate[0] > 0 else "left"
        vertical_direction = "upwards" if translate[1] < 0 else "downwards"
        if scale > 1:
            size_change = "enlarges in size"
        elif scale < 1:
            size_change = "reduces in size"
        else:
            size_change = "remains the same size"
        return horizontal_direction, vertical_direction, size_change


class SimpleLinearTrajectory(BaseTrajectory):
    def transform(self, img):
        img = TF.affine(
            img,
            angle=self.angle,
            translate=list(self.translate),
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        horizontal_direction, vertical_direction, size_change = self.describe_movement(
            self.translate, self.scale
        )
        transformation_caption = (
            f"The digit {self.digit_label} consistently moves {horizontal_direction} by {abs(self.translate[0]):.1f} pixels and "
            f"{vertical_direction} by {abs(self.translate[1]):.1f} pixels, rotates by {self.angle:.1f} degrees, and {size_change}."
        )

        return img, transformation_caption


class BouncingTrajectory(BaseTrajectory):
    def transform(self, img):
        new_position_x = self.position[0] + self.translate[0]
        new_position_y = self.position[1] + self.translate[1]

        # Check bounds
        if new_position_x <= -self.padding[0] or new_position_x >= self.padding[2]:
            self.translate = (-self.translate[0], self.translate[1])
        if new_position_y <= -self.padding[1] or new_position_y >= self.padding[3]:
            self.translate = (self.translate[0], -self.translate[1])

        self.position = (
            self.position[0] + self.translate[0],
            self.position[1] + self.translate[1],
        )

        img = TF.affine(
            img,
            angle=self.angle,
            translate=list(self.translate),
            scale=self.scale,
            shear=self.shear,
            **self.kwargs,
        )

        horizontal_direction, vertical_direction, size_change = self.describe_movement(
            self.translate, self.scale
        )
        transformation_caption = (
            f"The digit {self.digit_label} moves {horizontal_direction} by {abs(self.translate[0]):.1f} pixels and "
            f"{vertical_direction} by {abs(self.translate[1]):.1f} pixels, rotates by {self.angle:.1f} degrees, and {size_change}."
        )

        return img, transformation_caption


class RandomTrajectory(BaseTrajectory):
    def transform(self, img):
        # Get random values for each transform
        angle = random.uniform(*self.affine_params.angle)
        translate = (
            random.randint(*self.affine_params.translate[0]),
            random.randint(*self.affine_params.translate[1]),
        )
        scale = random.uniform(*self.affine_params.scale)
        shear = random.uniform(*self.affine_params.shear)

        new_position_x = self.position[0] + translate[0]
        new_position_y = self.position[1] + translate[1]

        # Check bounds
        if new_position_x <= -self.padding[0] or new_position_x >= self.padding[2]:
            translate = (-translate[0], translate[1])
        if new_position_y <= -self.padding[1] or new_position_y >= self.padding[3]:
            translate = (translate[0], -translate[1])

        self.position = (
            self.position[0] + translate[0],
            self.position[1] + translate[1],
        )

        img = TF.affine(
            img,
            angle=angle,
            translate=list(translate),
            scale=scale,
            shear=shear,
            **self.kwargs,
        )

        horizontal_direction, vertical_direction, size_change = self.describe_movement(
            translate, scale
        )
        transformation_caption = (
            f"The digit {self.digit_label} moves {horizontal_direction} by {abs(translate[0]):.1f} pixels and "
            f"{vertical_direction} by {abs(translate[1]):.1f} pixels, rotates by {angle:.1f} degrees, and {size_change}."
        )

        return img, transformation_caption
