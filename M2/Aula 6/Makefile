CC = mpicc
CFLAGS = -Wall 
LDFLAGS =

SRC = Ex1.c Ex2.c Ex3.c
OBJ = $(SRC:.c=.o)
TARGET = Ex1 Ex2 Ex3
all: $(TARGET)

# Compile source files to object files
%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

# Link object files to create executables
%: %.o
	$(CC) $(LDFLAGS) $< -o $@

clean:
	rm -f $(OBJS) $(PROGRAMS)