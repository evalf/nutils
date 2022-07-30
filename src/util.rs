pub trait ReplaceNthIter: Iterator + Sized {
    /// Replaces the nth item of the iterator with the given item.
    fn replace_nth(self, index: usize, item: Self::Item) -> ReplaceNth<Self>;
}

impl<Iter: Iterator> ReplaceNthIter for Iter {
    fn replace_nth(self, index: usize, item: Iter::Item) -> ReplaceNth<Self> {
        ReplaceNth(self, 0, index, item)
    }
}

pub struct ReplaceNth<Iter: Iterator>(Iter, usize, usize, Iter::Item);

impl<Iter: Iterator> Iterator for ReplaceNth<Iter> {
    type Item = Iter::Item;

    fn next(&mut self) -> Option<Self::Item> {
        self.0.next().map(|mut value| {
            if self.1 == self.2 {
                std::mem::swap(&mut self.3, &mut value);
            }
            self.1 += 1;
            value
        })
    }
}

pub trait SkipNthIter: Iterator + Sized {
    /// Skips the nth item of the iterator.
    fn skip_nth(self, index: usize) -> SkipNth<Self>;
}

impl<Iter: Iterator> SkipNthIter for Iter {
    fn skip_nth(self, index: usize) -> SkipNth<Self> {
        SkipNth(self, 0, index)
    }
}

pub struct SkipNth<Iter: Iterator>(Iter, usize, usize);

impl<Iter: Iterator> Iterator for SkipNth<Iter> {
    type Item = Iter::Item;

    fn next(&mut self) -> Option<Self::Item> {
        if let Some(value) = self.0.next() {
            let next = if self.1 == self.2 {
                self.0.next()
            } else {
                Some(value)
            };
            self.1 += 1;
            next
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn replace_nth() {
        let mut a = ['a', 'b', 'c', 'd'].into_iter().replace_nth(1, 'd');
        assert_eq!(a.next(), Some('a'));
        assert_eq!(a.next(), Some('d')); // replaced
        assert_eq!(a.next(), Some('c'));
        assert_eq!(a.next(), Some('d'));
        assert_eq!(a.next(), None);
    }
}
