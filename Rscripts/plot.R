#! /mnt/petrelfs/chenzhi/software/R/bin Rscript
# # Library
# library(ggplot2)
# library(dplyr)
# library(hrbrthemes)

# # Create data
# set.seed(1000)
# data <- data.frame(
#   x=LETTERS[1:26], 
#   y=abs(rnorm(26))
# )
 
# # Reorder the data
# data <- data %>%
#   arrange(y) %>%
#   mutate(x=factor(x,x))

# print(data)
  
# # Plot
# extrafont::loadfonts()
# p <- ggplot(data, aes(x=x, y=y)) +
#   geom_segment(
#     aes(x=x, xend=x, y=0, yend=y), 
#     color=ifelse(data$x %in% c("A","D"), "orange", "grey"), 
#     size=ifelse(data$x %in% c("A","D"), 1.3, 0.7)
#   ) +
#   geom_point(
#     color=ifelse(data$x %in% c("A","D"), "orange", "grey"), 
#     size=ifelse(data$x %in% c("A","D"), 5, 2)
#   ) +
#   coord_flip() +
#   theme(
#     legend.position="none"
#   ) +
#   xlab("") +
#   ylab("Value of Y") +
#   ggtitle("How did groups A and D perform?")

# # Add annotation
# p + annotate("text", x=grep("D", data$x), y=data$y[which(data$x=="D")]*1.2, 
#            label="Group D is very impressive", 
#            color="orange", size=4 , angle=0, fontface="bold", hjust=0) + 
  
#     annotate("text", x = grep("A", data$x), y = data$y[which(data$x=="A")]*1.2, 
#            label = paste("Group A is not too bad\n (val=",data$y[which(data$x=="A")] %>% round(2),")",sep="" ) , 
#            color="orange", size=4 , angle=0, fontface="bold", hjust=0) 

library(conflicted)
library(ggtext)
library(ragg)
library(tidyverse)
library(ggplot2)
library(rcartocolor)
library(here)

conflicts_prefer(dplyr::filter())

# Set ggplot theme
theme_set(theme_minimal(base_family = "Atlantis", base_size = 13))
theme_update(
  plot.margin = margin(25, 15, 15, 25),
  plot.background = element_rect(color = "#FFFCFC", fill = "#FFFCFC"),
  panel.grid.major.x = element_line(color = "grey94"),
  panel.grid.major.y = element_blank(),
  panel.grid.minor = element_blank(),
  axis.text = element_text(family = "Hydrophilia Iced"),
  axis.text.x = element_text(color = "grey40"),
  axis.text.y = element_blank(),
  axis.title = element_blank(),
  axis.ticks = element_blank(),
  legend.position = c(.07, .31),
  legend.title = element_text(
    color = "grey40",
    family = "Overpass",
    angle = 90,
    hjust = .5
  ),
  legend.text = element_text(
    color = "grey40",
    family = "Hydrophilia Iced",
    size = 12
  ),
  legend.box = "horizontal",
  legend.box.just = "bottom",
  legend.margin = margin(0, 0, 0, 0),
  legend.spacing = unit(.6, "lines"),
  plot.title = element_text(
    family = "Atlantis Headline",
    face = "bold",
    size = 17.45
  ),
  plot.subtitle = element_textbox_simple(
    family = "Overpass",
    color = "grey40",
    size = 10.8,
    lineheight = 1.3,
    margin = margin(t = 5, b = 30)
  ),
  plot.caption = element_text(
    family = "Overpass",
    color = "grey55",
    size = 10.5,
    margin = margin(t = 20, b = 0, r = 15)
  )
)

df_records <- readr::read_csv('https://raw.githubusercontent.com/rfordatascience/tidytuesday/master/data/2021/2021-05-25/records.csv')

print(df_records)

df_rank <-
  df_records %>%
  filter(type == "Three Lap") %>%
  group_by(track) %>%
  filter(time == min(time)) %>%
  ungroup %>%
  arrange(- time) %>%
  mutate(track = fct_inorder(track))

df_records_three <-
  df_records %>%
  filter(type == "Three Lap") %>%
  mutate(year = lubridate::year(date)) %>%
  mutate(track = factor(track, levels = levels(df_rank$track)))

df_connect <-
  df_records_three %>%
  group_by(track, type, shortcut) %>%
  summarize(no = min(time), yes = max(time)) %>%
  pivot_longer(
    cols = - c(track, type, shortcut),
    names_to = "record",
    values_to = "time"
  ) %>%
  # The line goes from the first record without shortcut (the slower)
  # to the most recent record with shortcut (the fastest)
  filter((shortcut == "No" & record == "no") | (shortcut == "Yes" & record == "yes")) %>%
  pivot_wider(id_cols = c(track), values_from = time, names_from = record)

df_longdist <-
  df_records_three %>%
  filter(shortcut == "No") %>%
  group_by(track) %>%
  filter(time == min(time) | time == max(time)) %>%
  mutate(group = if_else(time == min(time), "min", "max")) %>%
  group_by(track, group) %>%
  arrange(time) %>%
  slice(1) %>%
  group_by(track) %>%
  mutate(year = max(year)) %>%
  pivot_wider(id_cols = c(track, year), values_from = time, names_from = group) %>%
  mutate(diff = max - min)

df_shortcut <-
  df_records_three %>%
  filter(shortcut == "Yes") %>%
  group_by(track) %>%
  filter(time == min(time) | time == max(time)) %>%
  mutate(group = if_else(time == min(time), "min", "max")) %>%
  group_by(track, group) %>%
  arrange(time) %>%
  slice(1) %>%
  group_by(track) %>%
  mutate(year = max(year)) %>%
  pivot_wider(id_cols = c(track, year), values_from = time, names_from = group) %>%
  mutate(diff = max - min)

# This dataset `df_shorcut` and the aesthetics `aes(min, track)` are used
# in the `geom_*` calls when they are not explicitly specified.
p <- df_shortcut %>%
  ggplot(aes(min, track)) +
  # Dotted line connection shortcut yes/no
  # This geom uses `df_connect` instead of `df_shorcut` because it is being
  # explicitly overridden
  geom_linerange(
    data = df_connect,
    aes(xmin = yes, xmax = no, y = track),
    inherit.aes = FALSE,
    color = "grey75",
    linetype = "11" # dotted line
  ) +
  # Segment when shortcut==yes
  # When the `data` argument is missing in the `geom_*` function
  geom_linerange(aes(xmin = min, xmax = max, color = diff), size = 2) +
  # Segment when shortcut==no. Overlapped lineranges.
  geom_linerange(data = df_longdist, aes(xmin = min, xmax = max, color = diff), size = 2) +
  geom_linerange(data = df_longdist, aes(xmin = min, xmax = max), color = "#FFFCFC", size = .8)

p <- p +
  # Point when shortcut==yes – first record
  geom_point(aes(x = max), size = 7, color = "#FFFCFC", fill = "grey65", shape = 21, stroke = .7) +
  # Point when shortcut==yes – latest record.
  geom_point(aes(fill = year), size = 7, color = "#FFFCFC", shape = 21, stroke = .7) +
  # Point when shortcut==no – first record.
  geom_point(data = df_longdist, aes(fill = year), size = 5.6, shape = 21,
             color = "#FFFCFC", stroke = .5) +
  geom_point(data = df_longdist, size = 3, color = "#FFFCFC") +
  # Point when shortcut==no – latest record
  geom_point(data = df_longdist, aes(x = max), size = 5.6, shape = 21,
             fill = "grey65", color = "#FFFCFC", stroke = .5) +
  geom_point(data = df_longdist, aes(x = max), size = 3, color = "#FFFCFC")

p <- p +
  ## labels tracks
  geom_label(aes(label = track), family = "Atlantis", size = 6.6, hjust = 1, nudge_x = -7,
             label.size = 0, fill = "#FFFCFC") +
  geom_label(data = filter(df_longdist, !track %in% unique(df_shortcut$track)), 
             aes(label = track), family = "Atlantis", size = 6.6, hjust = 1, nudge_x = -7,
             label.size = 0, fill = "#FFFCFC") +
  ## labels dots when shortcut==yes
  geom_text(data = filter(df_shortcut, track == "Wario Stadium"),
             aes(label = "Most recent record\nwith shortcuts"),
             family = "Overpass", size = 3.5, color = "#4a5a7b",
             lineheight = .8, vjust = 0, nudge_y = .4) +
  geom_text(data = filter(df_shortcut, track == "Wario Stadium"),
             aes(x = max, label = "First record\nwith shortcuts"),
             family = "Overpass", size = 3.5, color = "grey50",
             lineheight = .8, vjust = 0, nudge_y = .4) +
  ## labels dots when shortcut==no
  geom_text(data = filter(df_longdist, track == "Wario Stadium"),
             aes(label = "Most recent record\nw/o shortcuts"), 
             family = "Overpass", size = 3.5, color = "#4a5a7b", lineheight = .8,
             vjust = 0, nudge_x = -7, nudge_y = .4) +
  geom_text(data = filter(df_longdist, track == "Wario Stadium"),
             aes(x = max, label = "First record\nw/o shortcuts"), 
             family = "Overpass", size = 3.5, color = "grey50", lineheight = .8,
             vjust = 0, nudge_x = 7, nudge_y = .4)

p <- p +
  # Extend horizontal axis so trackl labels fit
  coord_cartesian(xlim = c(-60, 400)) +
  scale_x_continuous(
    breaks = seq(0, 400, by = 50),
    # Add 'seconds' label only to first axis tick
    labels = function(x) ifelse(x == 0, paste(x, "seconds"), paste(x)),
    sec.axis = dup_axis(), # Add axis ticks and labels both on top and bottom.
    expand = c(.02, .02)
  ) +
  scale_y_discrete(expand = c(.07, .07)) +
  scale_fill_gradient(low = "#b4d1d2", high = "#242c3c", name = "Year of Record") +
  rcartocolor::scale_color_carto_c(
    palette = "RedOr",
    limits = c(0, 250),
    name = "Time difference between first and most recent record"
  )

p <- p +
  guides(
    fill = guide_legend(title.position = "left"),
    color = guide_colorbar(
      barwidth = unit(.45, "lines"),
      barheight = unit(22, "lines"),
      title.position = "left"
    )
  ) +
  labs(
    title = "Let's-a-Go!  You  May  Still  Have  Chances  to  Grab  a  New  World  Record  for  Mario  Kart  64",
    subtitle = "Most world records for Mario Kart 64 were achieved pretty recently (13 in 2020, 10 in 2021). On several tracks, the players considerably improved the time needed to complete three laps when they used shortcuts (*Choco Mountain*, *D.K.'s Jungle Parkway*, *Frappe Snowland*, *Luigi Raceway*, *Rainbow Road*, *Royal Raceway*, *Toad's Turnpike*, *Wario Stadium*, and *Yoshi Valley*). Actually, for three out of these tracks the previous records were more than halved since 2020 (*Luigi Raceway*, *Rainbow Road*, and *Toad's Turnpike*). Four other tracks still have no records for races with shortcuts (*Moo Moo Farm*, *Koopa Troopa Beach*, *Banshee Boardwalk*, and *Bowser's Castle*). Are there none or did nobody find them yet? Pretty unrealistic given the fact that since more than 24 years the game is played all around the world—but maybe you're able to find one and obtain a new world record?", # nolint: line_length_linter.
    caption = "Visualization: Cédric Scherer  •  Data: mkwrs.com/mk64"
  )

# options(ragg.max_dim = 1e5) # nolint: commented_code_linter.

ggsave(
  filename = "test.png", width = 18, height = 11.5, dpi = 600, limitsize = FALSE
)